"""A kripto megjelenítési fájljai (5. fázis, E4).

Saját fájlokba ír (`crypto/…`, `record-crypto/…`): a napi adatfrissítés az app
élesítésétől függetlenül fut, és a mostani felület a kriptót nem ismeri. Ha a
kripto a meglévő listákba kerülne, jelölés nélkül keveredne a részvényekkel.
A papír-csomag (`instruments/CZ006xx.json`) és a munkaasztal idősora ugyanazzal
az építővel készül, mint a részvényeké — a felület ugyanúgy olvassa.

Futtatás:
    uv run python -m pipeline.crypto.publish
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pipeline import log as logging_setup
from pipeline.assetspec import AssetSpec
from pipeline.assetspec import crypto as crypto_spec
from pipeline.config import load_settings
from pipeline.crypto.resolve import load_forecasts, load_outcomes
from pipeline.features.run import _read_table
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.publish.record import build_record
from pipeline.publish.run import (
    DISPLAY_BUCKET,
    _dumps,
    build_arena,
    build_history,
    build_index,
    build_instrument,
    screener_row,
)
from pipeline.universe import active_on

log = logging_setup.get_logger(__name__)


def _latest(
    day: str,
    today: pd.DataFrame,
    outcomes: pd.DataFrame,
    universe: int,
    first_live: str | None,
    now: datetime,
    spec: AssetSpec,
) -> dict[str, object]:
    resolved: dict[str, object] = {"count": 0}
    if not outcomes.empty and "resolved_at" in outcomes:
        stamp = pd.to_datetime(outcomes["resolved_at"])
        recent = outcomes[stamp == stamp.max()]
        resolved = {
            "count": len(recent),
            "hits": int(recent["hit"].sum()),
            "baseline_hits": int(recent["baseline_hit"].sum()),
            "on": str(recent["target_session"].max()),
        }
    regime = None
    if not today.empty and "regime" in today and not today["regime"].dropna().empty:
        regime = str(today["regime"].dropna().mode().iloc[0])
    return {
        "generated_at": now.isoformat(),
        "session": day,
        "calendar": spec.calendar,
        "model": f"{spec.family} {spec.version}",
        "regime": regime,
        "instruments_with_forecast": int(today["instrument_id"].nunique()) if not today.empty else 0,
        "universe": universe,
        "resolved": resolved,
        # Az élő rekord innen számít (docs/kripto-modell.md, 6.).
        "live_from": first_live,
    }


def _in_scope(table: pd.DataFrame, scope: str) -> pd.DataFrame:
    """A régi táblákban nincs `scope` oszlop: azok egészében a fő családé."""
    if table.empty or "scope" not in table:
        return table
    return table[table["scope"] == scope]


def run(
    storage: Storage, now: datetime, dry_run: bool = False, spec: AssetSpec | None = None
) -> dict[str, object]:
    """Az eszközosztály megjelenítési fájljai; alapból a kriptóé (`pipeline.assetspec`)."""
    spec = spec or crypto_spec()
    prefix, export_prefix, calendar = spec.display_prefix, spec.export_prefix, spec.calendar
    last = spec.last_day(now)
    universe = active_on(spec.load_universe(), now.astimezone(UTC).date())
    # A felület a megjelenítési idősort mutatja (kötvényen a hozamot, %-ban).
    load_display = spec.load_display_prices or spec.load_prices
    prices = load_display(storage, list(range(spec.history_start_year, last.year + 1)))
    years = list(range(spec.first_year, last.year + 1))
    forecasts = load_forecasts(storage, years, spec.forecasts_prefix)
    outcomes = load_outcomes(storage, years, spec.outcomes_prefix)

    if forecasts.empty:
        day, today, first_live = str(last), pd.DataFrame(columns=["instrument_id"]), None
    else:
        newest = max(forecasts["session"])
        day, today, first_live = (
            str(newest),
            forecasts[forecasts["session"] == newest],
            str(min(forecasts["session"])),
        )

    files: list[tuple[str, bytes, str]] = []

    def add(path: str, payload: object) -> None:
        files.append((path, _dumps(payload), "application/json"))

    # A fő rekord a külön mért szegmens nélkül (deviza: keresztpárok, 7. fejezet).
    main_outcomes, _ = spec.split(outcomes, universe)
    add(f"{prefix}/latest.json", _latest(day, today, main_outcomes, len(universe), first_live, now, spec))
    index = build_index(universe, today, prices)
    add(f"{prefix}/instruments.json", [{**row, "calendar": calendar} for row in index])

    live = _read_table(storage, spec.live_path)
    backtest = _read_table(storage, spec.backtest_path)
    live = live if live is not None else pd.DataFrame()
    backtest = backtest if backtest is not None else pd.DataFrame()
    evidence: dict[str, object] = {
        "generated_at": now.isoformat(),
        "calendar": calendar,
        "live_from": first_live,
        "live": build_arena(_in_scope(live, spec.scope)),
        "backtest": build_arena(_in_scope(backtest, spec.scope)),
    }
    if spec.cross_scope is not None:
        evidence["cross"] = {
            "live": build_arena(_in_scope(live, spec.cross_scope)),
            "backtest": build_arena(_in_scope(backtest, spec.cross_scope)),
        }
    add(f"{prefix}/evidence.json", evidence)

    existing = set() if dry_run else set(storage.list(DISPLAY_BUCKET, export_prefix))
    record, record_files = build_record(main_outcomes, forecasts, universe, existing, now, export_prefix)
    add(f"{prefix}/record.json", record)
    files.extend(record_files)

    by_price = {i: g for i, g in prices.groupby("instrument_id")}
    by_today = {i: g for i, g in today.groupby("instrument_id")} if not today.empty else {}
    by_outcome = {i: g for i, g in outcomes.groupby("instrument_id")} if not outcomes.empty else {}
    history_all = (
        forecasts if not forecasts.empty else pd.DataFrame(columns=["instrument_id", "session", "horizon"])
    )
    by_history = {i: g for i, g in history_all.groupby("instrument_id")}
    screener = []
    for row in universe.to_dict("records"):
        iid = str(row["instrument_id"])
        meta = {
            "id": iid,
            "ticker": row["ticker"],
            "name": row["name"],
            "asset_class": spec.name,
            "sector": None,
            "exchange_calendar": calendar,
        }
        bars = by_price.get(iid, prices.iloc[0:0])
        payload = build_instrument(
            meta=meta,
            prices=bars.tail(spec.history_days),
            forecasts=by_today.get(iid, pd.DataFrame()),
            history=by_history.get(iid, pd.DataFrame(columns=["session", "horizon"])),
            outcomes=by_outcome.get(iid, pd.DataFrame()),
            session=pd.Timestamp(day).date(),
        )
        add(f"instruments/{iid}.json", payload)
        if not bars.empty:
            add(f"history/{iid}.json", build_history(bars))
        screener.append(
            screener_row(
                {k: meta[k] for k in ("id", "ticker", "name", "sector", "asset_class")},
                bars,
                by_today.get(iid, pd.DataFrame()),
                by_outcome.get(iid, pd.DataFrame(columns=["brier", "baseline_brier"])),
            )
        )
    add(f"{prefix}/screener.json", {"session": day, "rows": screener})

    if dry_run:
        return {"files": len(files), "status": "dry_run"}
    storage.ensure_private_bucket(DISPLAY_BUCKET)
    from pipeline.publish.manifest import ChangedOnly

    # Eszközosztályonként saját manifeszt: a futások párhuzamosan írnak, és
    # egy közös fájlt felülírnák egymásnak.
    uploader = ChangedOnly(storage, DISPLAY_BUCKET, f"{prefix}/.manifest.json")
    try:
        for path, blob, content_type in files:
            uploader.put(path, blob, content_type)
    finally:
        uploader.finish()
    log.info("asset_publish_done", asset=spec.name, files=len(files), day=day)
    return {"files": len(files), "day": day, "status": "published"}


def main(argv: list[str] | None = None, spec: AssetSpec | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calyze kripto megjelenítési fájlok")
    parser.add_argument("--local", type=Path)
    parser.add_argument("--dry-run", action="store_true")
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
    from pipeline.runstate import gated

    spec = spec or crypto_spec()
    result = gated(
        storage, f"publish-{spec.name}", lambda: run(storage, datetime.now(UTC), args.dry_run, spec)
    )
    print(json.dumps(result if result is not None else {"status": "skipped"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
