"""A kötvény-hozamgörbe napi letöltése (`docs/kotveny.md`, 1–3.).

A pénzügyminisztérium évenkénti CSV-jéből. Az idősor a hozam %-ban (a négy ár
ugyanaz, volumen nincs); a meredekség két oszlop különbsége. Külön fájlokba ír
(`prices_bonds_daily/`). A hiányzó kötvénynap hiányként látszik, nem pótoljuk.

Futtatás:
    uv run python -m pipeline.ingest.bonds --mode daily
    uv run python -m pipeline.ingest.bonds --mode backfill --local data
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline import log as logging_setup
from pipeline.bonds.calendar import bond_days, is_bond_day, last_bond_day
from pipeline.config import RAW_BUCKET, load_settings
from pipeline.ingest.partitions import read_partition, upsert, write_partitions
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.ingest.validate import mark_quality
from pipeline.universe import active_on, load_bonds_universe

log = logging_setup.get_logger(__name__)

BONDS_PRICES_PREFIX = "prices_bonds_daily"
BONDS_HISTORY_START = date(1990, 1, 2)
URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
    "{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv"
)
#: A reálhozam-görbe (TIPS) 2003-tól; ugyanabból a 15:30-as felvételből készül.
REAL_FROM_YEAR = 2003
#: A reálhozam oszlopai ezzel az előtaggal kerülnek a görbébe (a minisztérium
#: „5 YR”-nek hívja, a nominálisat „5 Yr”-nek: a kis-nagybetű nem elég jel).
REAL_PREFIX = "Real "
#: Azonosítható kérés: a forrás így tudja, ki kéri (nem rejtőzködünk).
HEADERS = {"User-Agent": "Calyze research pipeline (github.com/sigmundsabo/calyze-data)"}


class TreasuryError(RuntimeError):
    """A pénzügyminisztérium nem adott használható választ."""


def fetch_year(year: int, timeout: float = 60, kind: str = "daily_treasury_yield_curve") -> pd.DataFrame:
    for attempt in range(3):
        try:
            r = requests.get(URL.format(year=year, kind=kind), headers=HEADERS, timeout=timeout)
        except requests.RequestException as error:
            if attempt == 2:
                raise TreasuryError(type(error).__name__) from error
            continue
        if r.status_code == 200 and r.text.startswith("Date"):
            frame = pd.read_csv(io.StringIO(r.text))
            frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y").dt.date
            return frame
    raise TreasuryError(f"HTTP {r.status_code}")


def fetch_real_year(year: int, timeout: float = 60) -> pd.DataFrame:
    """A reálhozam-görbe egy éve, előtaggal; 2003 előtt üres."""
    if year < REAL_FROM_YEAR:
        return pd.DataFrame(columns=["Date"])
    frame = fetch_year(year, timeout, kind="daily_treasury_real_yield_curve")
    return frame.rename(columns={c: f"{REAL_PREFIX}{c}" for c in frame.columns if c != "Date"})


def fetch_curve(year: int) -> pd.DataFrame:
    """A nominális és a reál görbe egy éve, dátum szerint összefésülve.

    A reálgörbe hibája nem állíthatja meg a nominálisat: akkor azok a sorok
    aznap hiányoznak, és a hiány látszik.
    """
    nominal = fetch_year(year)
    try:
        real = fetch_real_year(year)
    except TreasuryError as error:
        log.warning("bonds_real_curve_failed", year=year, error=str(error))
        return nominal
    if real.empty:
        return nominal
    return nominal.merge(real, on="Date", how="left")


def series(curve: pd.DataFrame, universe: pd.DataFrame, fetched_at: datetime) -> pd.DataFrame:
    """Az idősorok a kanonikus ár-sémában (hozam %-ban; a meredekség különbség)."""
    rows = []
    for r in universe.itertuples():
        spec = str(r.source_symbol)
        if "-" in spec:
            long, short = spec.split("-")
            if long not in curve or short not in curve:
                continue
            value = curve[long] - curve[short]
        else:
            if spec not in curve:
                continue
            value = curve[spec]
        part = pd.DataFrame({"date": curve["Date"], "v": pd.to_numeric(value, errors="coerce")}).dropna()
        rows.append(part.assign(instrument_id=r.instrument_id))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    frame = pd.DataFrame(
        {
            "instrument_id": out["instrument_id"],
            "date": out["date"],
            "open": out["v"],
            "high": out["v"],
            "low": out["v"],
            "close": out["v"],
            "adj_close": out["v"],
            "volume": np.nan,
            "source": "treasury",
            "fetched_at": pd.Timestamp(fetched_at),
        }
    )
    # A `mark_quality` a nem pozitív zárót gyanúsnak jelölné; a meredekség és
    # a nulla közeli rövid hozam viszont valódi érték, ezért itt mindig „ok”.
    return (
        mark_quality(frame).assign(quality="ok").sort_values(["instrument_id", "date"]).reset_index(drop=True)
    )


def run(mode: str, storage: Storage, now: datetime) -> dict[str, object]:
    fetched_at = now.astimezone(UTC)
    last = last_bond_day(now)
    universe = active_on(load_bonds_universe(), fetched_at.date())
    years = (
        range(BONDS_HISTORY_START.year, last.year + 1)
        if mode == "backfill"
        else range(last.year - 1, last.year + 1)
    )
    curve = pd.concat([fetch_curve(y) for y in years], ignore_index=True)
    curve = curve[curve["Date"] <= last]
    # A naptár dönti el, mi kötvénynap (docs/kotveny.md, 2.): a minisztérium
    # néhány ünnepen (nagypéntek NFP-vel, péntekre hozott ünnep) is közölt.
    off_calendar = sorted(d.isoformat() for d in set(curve["Date"]) if not is_bond_day(d))
    curve = curve[[is_bond_day(d) for d in curve["Date"]]]
    fresh = series(curve, universe, fetched_at)
    if fresh.empty:
        raise TreasuryError("a görbéből egyetlen idősor sem állt elő")
    start = min(curve["Date"]) if mode == "backfill" else date(last.year - 1, 1, 1)
    missing = [d.isoformat() for d in bond_days(start, last) if d not in set(curve["Date"])]
    storage.ensure_private_bucket(RAW_BUCKET)
    if mode == "backfill":
        written = write_partitions(storage, fresh, BONDS_PRICES_PREFIX)
    else:
        written = []
        for year in sorted({d.year for d in fresh["date"]}):
            current = read_partition(storage, year, BONDS_PRICES_PREFIX)
            merged = upsert(current, fresh[[d.year == year for d in fresh["date"]]])
            if not merged.empty:
                written += write_partitions(storage, merged, BONDS_PRICES_PREFIX)
    summary: dict[str, object] = {
        "asset_class": "bond",
        "mode": mode,
        "run_at": fetched_at.isoformat(timespec="seconds"),
        "last_day": last.isoformat(),
        "series": int(fresh["instrument_id"].nunique()),
        "rows": len(fresh),
        "missing_bond_days": missing,
        "off_calendar_days": off_calendar,
        "partitions_written": sorted(set(written)),
        "source": "treasury",
    }
    storage.upload(
        RAW_BUCKET,
        f"runs/ingest-bonds/{last.isoformat()}-{mode}.json",
        json.dumps(summary, indent=2).encode(),
        "application/json",
    )
    log.info(
        "bonds_ingest_done",
        **{k: v for k, v in summary.items() if k not in ("missing_bond_days", "off_calendar_days")},
        gaps=missing[-10:],
        off_calendar=off_calendar[-10:],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calyze kötvény-letöltés (US Treasury)")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--local", type=Path)
    args = parser.parse_args(argv)
    logging_setup.configure()
    if args.local:
        storage: Storage = LocalStorage(args.local)
    else:
        settings = load_settings()
        if not settings.supabase_url or not settings.supabase_secret_key:
            log.error("storage_not_configured")
            return 2
        storage = SupabaseStorage(settings.supabase_url, settings.supabase_secret_key)
    try:
        summary = run(args.mode, storage, datetime.now(UTC))
    except TreasuryError as error:
        log.error("bonds_ingest_failed", error=str(error))
        return 1
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("missing_bond_days", "off_calendar_days")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
