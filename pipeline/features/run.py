"""Feature-számítás az árfolyam-letöltés után (spec/05, 5. fejezet: az `ingest` része).

**A feature-öket nem tároljuk teljes hosszban.** Az árfolyamból és az
eseménylistából determinisztikusan újraszámolhatók (`build_features`, kb.
2,5 perc a teljes múltra), a tárolásuk viszont a teljes múltra kb. 420 MB
lenne — az ingyenes tár (1 GB) majdnem felét és a kimenő forgalom nagy részét
vinné el. A tanítás és a becslés (M2) ugyanezt a függvényt hívja, így
nincs két, egymástól elcsúszó változat. Tárolva csak ez marad:

- `features/latest.parquet` — az utolsó 10 session feature-jei (ellenőrzéshez);
- `regime_daily.parquet`, `macro_daily.parquet` — kicsik, és a felület is használja;
- `basket_daily.parquet` — a rezsim piaci kosara teljes hosszban (a bővülő
  percentilishez a teljes múlt kell, a napi futás viszont csak 3 évet tölt le).

A napló és a futási összefoglaló csak darabszámot tartalmaz; a rezsim-címke
és minden érték a privát tárban marad (a repó és a naplója publikus).

Futtatás:
    uv run python -m pipeline.features.run --mode daily
    uv run python -m pipeline.features.run --mode backfill --local data
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from pipeline import log as logging_setup
from pipeline.calendar import last_closed_session, sessions_back
from pipeline.config import DAILY_WINDOW_SESSIONS, HISTORY_START, RAW_BUCKET, load_settings
from pipeline.corporate import total_return_prices
from pipeline.features.regime import CROSS_MARKET, compute_regime
from pipeline.features.technical import add_cross_sectional, compute_technical
from pipeline.ingest.partitions import ACTIONS_PATH, from_parquet, read_partition, to_parquet, upsert
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.macro import align_to_sessions, fetch_all
from pipeline.universe import active_on, load_universe

log = logging_setup.get_logger(__name__)

LATEST_PATH = "features/latest.parquet"
BASKET_PATH = "basket_daily.parquet"
REGIME_PATH = "regime_daily.parquet"
MACRO_PATH = "macro_daily.parquet"
DAILY_LOOKBACK_YEARS = 3


def _write_table(storage: Storage, path: str, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    storage.upload(RAW_BUCKET, path, to_parquet(frame, table.schema), "application/octet-stream")


def _read_table(storage: Storage, path: str) -> pd.DataFrame | None:
    data = storage.download(RAW_BUCKET, path)
    return from_parquet(data) if data is not None else None


def _load_prices(storage: Storage, years: list[int]) -> pd.DataFrame:
    frames = [read_partition(storage, y) for y in years]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _macro(api_key: str | None, start: date, end: date, required: bool) -> pd.DataFrame | None:
    if not api_key:
        if required:
            raise RuntimeError("Nincs FRED_API_KEY — a makro-feature-ök nélkül a futás nem teljes.")
        log.warning("macro_skipped", reason="nincs FRED_API_KEY (helyi próbafuttatás)")
        return None
    return align_to_sessions(fetch_all(api_key, start, end), start, end)


def add_macro(features: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    """A makro-feature-ök (már egy session késleltetéssel) a dátumhoz kötve."""
    cols = ["vix", "vix_chg_20", "yield_10y", "yield_curve_10y2y", "dollar_ret_20"]
    if macro is None:
        return features.assign(**{c: np.nan for c in cols})
    m = macro.copy()
    m["vix_chg_20"] = np.log(m["vix"] / m["vix"].shift(20))
    m["dollar_ret_20"] = np.log(m["dollar_index"] / m["dollar_index"].shift(20))
    return features.join(m[cols], on="date")


def build_features(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    universe: pd.DataFrame,
    regime: pd.DataFrame,
    macro: pd.DataFrame | None,
) -> pd.DataFrame:
    """A teljes feature-tábla: technikai + keresztmetszeti + makro + rezsim."""
    priced = prices.assign(tr_close=total_return_prices(prices, actions))
    feats = compute_technical(priced)
    feats = add_cross_sectional(feats, universe)
    feats = add_macro(feats, macro)
    return feats.join(regime.set_index("date")[["stress", "regime"]], on="date")


def run(
    mode: str, storage: Storage, now: datetime, fred_key: str | None, macro_required: bool
) -> dict[str, object]:
    last = last_closed_session(now)
    # A mutatók csak a becsléshez kellenek: ha a napra már van becslés-csomag,
    # a napi újrapróba nem tölti le újra a három év árfolyamot (J2).
    if mode == "daily":
        from pipeline.forecast.run import package_exists

        if package_exists(storage, last):
            log.info("features_skipped", reason="forecast_already_saved", session=str(last))
            return {"status": "skipped", "session": str(last)}
    universe = active_on(load_universe(), now.astimezone(UTC).date())
    etf_ids = set(universe.loc[universe["ticker"].isin(CROSS_MARKET), "instrument_id"])
    actions = _read_table(storage, ACTIONS_PATH)
    if actions is None:
        actions = pd.DataFrame(columns=["instrument_id", "date", "dividend", "split_ratio"])

    if mode == "backfill":
        years = list(range(HISTORY_START.year, last.year + 1))
    else:
        years = list(range(last.year - DAILY_LOOKBACK_YEARS + 1, last.year + 1))
    prices = _load_prices(storage, years)
    if prices.empty:
        raise RuntimeError("Nincs árfolyam a privát tárban — előbb az ingest fusson le.")

    fresh_basket = prices[prices["instrument_id"].isin(etf_ids)]
    stored = None if mode == "backfill" else _read_table(storage, BASKET_PATH)
    basket = upsert(stored, fresh_basket) if stored is not None else fresh_basket.reset_index(drop=True)
    regime = compute_regime(basket.assign(tr_close=total_return_prices(basket, actions)), universe)
    macro = _macro(fred_key, HISTORY_START, last, macro_required)

    feats = build_features(prices, actions, universe, regime, macro)
    window_start = sessions_back(last, DAILY_WINDOW_SESSIONS)[0]
    latest = feats[feats["date"] >= window_start].reset_index(drop=True)

    _write_table(storage, LATEST_PATH, latest)
    _write_table(storage, BASKET_PATH, basket.reset_index(drop=True))
    _write_table(storage, REGIME_PATH, regime)
    if macro is not None:
        _write_table(storage, MACRO_PATH, macro.reset_index())

    labelled = regime.dropna(subset=["regime"])
    coverage = latest.groupby("instrument_id")["date"].max()
    summary: dict[str, object] = {
        "mode": mode,
        "run_at": now.astimezone(UTC).isoformat(timespec="seconds"),
        "last_session": last.isoformat(),
        "feature_rows_computed": len(feats),
        "feature_columns": len([c for c in feats.columns if c not in ("instrument_id", "date")]),
        "instruments": int(feats["instrument_id"].nunique()),
        "instruments_current": int((coverage == last).sum()),
        "macro": macro is not None,
        "regime_sessions_labelled": len(labelled),
    }
    # A privát összefoglalóba a rezsim-eloszlás is bekerül; a publikus naplóba nem.
    private = {**summary, "regime_counts": labelled["regime"].value_counts().to_dict()}
    storage.upload(
        RAW_BUCKET,
        f"runs/features/{last.isoformat()}-{mode}.json",
        json.dumps(private, indent=2, default=str).encode(),
        "application/json",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calyze feature-számítás")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--local", type=Path, help="helyi mappa a privát tár helyett (próbafuttatás)")
    args = parser.parse_args(argv)

    logging_setup.configure()
    settings = load_settings()
    now = datetime.now(UTC)
    if args.local:
        storage: Storage = LocalStorage(args.local)
    else:
        if not settings.supabase_url or not settings.supabase_secret_key:
            log.error("storage_not_configured")
            return 2
        storage = SupabaseStorage(settings.supabase_url, settings.supabase_secret_key)

    summary = run(args.mode, storage, now, settings.fred_api_key, macro_required=not args.local)
    log.info("features_done", **{k: v for k, v in summary.items() if k != "run_at"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
