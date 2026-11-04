"""A kripto- és deviza-becslések és -tézisek kiértékelése (5. fázis E4, 6. fázis F4).

Ugyanaz a lezárás, mint a részvényeknél (`pipeline.resolve.run.resolve_due`):
a becslés sora nem módosul, a kimenetel külön sorban keletkezik, egyszer. Az
eszközosztály saját táblákba ír (`outcomes-crypto/`, `outcomes-fx/` …), és a
mérési rekordjai saját családot alkotnak (`scope`), saját FDR-korrekcióval.
Melyik eszközosztály: a `pipeline.assetspec` leírása dönti el.

Futtatás:
    uv run python -m pipeline.crypto.resolve
    uv run python -m pipeline.fx.resolve
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
from pipeline.config import RAW_BUCKET, load_settings
from pipeline.features.run import _read_table, _write_table
from pipeline.ingest.partitions import from_parquet
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.journal import resolve as journal_resolve
from pipeline.resolve.run import live_arena, resolve_due

log = logging_setup.get_logger(__name__)

EMPTY_ACTIONS = pd.DataFrame(columns=["instrument_id", "date", "dividend", "split_ratio"])
# A kripto alapértékei (a régi hívók és a tesztek ezeket importálják).
OUTCOMES_PREFIX = "outcomes-crypto"
LIVE_PATH = "arena/crypto_live.parquet"
FIRST_YEAR = 2026


def outcomes_path(year: int, prefix: str = OUTCOMES_PREFIX) -> str:
    return f"{prefix}/year={year}.parquet"


def load_forecasts(storage: Storage, years: list[int], prefix: str = "forecasts-crypto") -> pd.DataFrame:
    frames = []
    for year in years:
        for path in storage.list(RAW_BUCKET, f"{prefix}/{year}"):
            blob = storage.download(RAW_BUCKET, path)
            if blob is not None:
                frames.append(from_parquet(blob))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_outcomes(storage: Storage, years: list[int], prefix: str = OUTCOMES_PREFIX) -> pd.DataFrame:
    frames = [f for f in (_read_table(storage, outcomes_path(y, prefix)) for y in years) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scoped_live_arena(outcomes: pd.DataFrame, scope: str) -> pd.DataFrame:
    """A részvényes élő rekord-építő, az eszközosztály saját családjaként."""
    arena = live_arena(outcomes)
    if arena.empty:
        return arena
    return arena.assign(scope=scope)


def crypto_live_arena(outcomes: pd.DataFrame) -> pd.DataFrame:
    return scoped_live_arena(outcomes, "crypto")


def run(
    storage: Storage,
    now: datetime,
    journal: journal_resolve.JournalStore | None = None,
    spec: AssetSpec | None = None,
) -> dict[str, object]:
    spec = spec or crypto_spec()
    last = spec.last_day(now)
    years = list(range(spec.first_year, last.year + 1))
    forecasts = load_forecasts(storage, years, spec.forecasts_prefix)
    prices = spec.load_prices(
        storage, list(range(max(spec.history_start_year, last.year - 1), last.year + 1))
    )

    summary: dict[str, object] = {"asset_class": spec.name, "last_day": last.isoformat()}
    if not forecasts.empty:
        known = load_outcomes(storage, years, spec.outcomes_prefix)
        done = set(known["forecast_id"]) if not known.empty else set()
        pending = forecasts[~forecasts["forecast_id"].isin(done)]
        fresh = resolve_due(pending, prices, EMPTY_ACTIONS, last, now.astimezone(UTC))
        outcomes = pd.concat([known, fresh], ignore_index=True) if not fresh.empty else known
        if not fresh.empty:
            for year in sorted({d.year for d in outcomes["session"]}):
                part = outcomes[[d.year == year for d in outcomes["session"]]].reset_index(drop=True)
                _write_table(storage, outcomes_path(year, spec.outcomes_prefix), part)
        main, cross = spec.split(outcomes, spec.load_universe())
        arena = scoped_live_arena(main, spec.scope)
        if spec.cross_scope is not None and not cross.empty:
            arena = pd.concat([arena, scoped_live_arena(cross, spec.cross_scope)], ignore_index=True)
        if not arena.empty:
            _write_table(storage, spec.live_path, arena)
        summary |= {
            "resolved_now": len(fresh),
            "resolved_total": len(outcomes),
            "open": int(len(forecasts) - len(outcomes)),
            "live_records": len(arena),
        }
    else:
        summary |= {"resolved_now": 0, "resolved_total": 0, "open": 0, "live_records": 0}

    # A tézisek: csak az eszközosztály saját árfolyamával és naptárával. A
    # tézisek hibája nem állíthatja meg a modell mérését (mint a részvényeknél).
    if journal is not None and spec.theses and not prices.empty:
        try:
            series = journal_resolve.as_series(prices.assign(tr=prices["close"]))
            journal_resolve.run(journal, series, forecasts, last, spec.calendar)
        except Exception as error:  # noqa: BLE001
            log.error("journal_resolve_failed", asset=spec.name, error=type(error).__name__)

    # A forgatókönyvek kötvényre is (nem tézis: a modell becslését rögzíti).
    if journal is not None and not prices.empty and not forecasts.empty:
        from pipeline.scenario.resolve import run_beside_journal

        series = journal_resolve.as_series(prices.assign(tr=prices["close"]))
        run_beside_journal(journal, storage, series, forecasts, last, f"{spec.display_prefix}/")

    storage.upload(
        RAW_BUCKET,
        f"runs/resolve-{spec.name}/{last.isoformat()}.json",
        json.dumps(summary, indent=2, default=str).encode(),
        "application/json",
    )
    log.info("asset_resolve_done", **summary)
    return summary


def main(argv: list[str] | None = None, spec: AssetSpec | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calyze kripto- és deviza-kiértékelés")
    parser.add_argument("--local", type=Path)
    args = parser.parse_args(argv)
    logging_setup.configure()
    settings = load_settings()
    journal: journal_resolve.JournalStore | None = None
    if args.local:
        storage: Storage = LocalStorage(args.local)
    elif settings.supabase_url and settings.supabase_secret_key:
        storage = SupabaseStorage(settings.supabase_url, settings.supabase_secret_key)
        journal = journal_resolve.SupabaseJournal(settings.supabase_url, settings.supabase_secret_key)
    else:
        log.error("storage_not_configured")
        return 2
    from pipeline.runstate import gated

    spec = spec or crypto_spec()
    result = gated(storage, f"resolve-{spec.name}", lambda: run(storage, datetime.now(UTC), journal, spec))
    print(json.dumps(result if result is not None else {"status": "skipped"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
