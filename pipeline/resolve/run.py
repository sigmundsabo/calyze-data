"""Kiértékelés: a lejárt becslések lezárása (spec/06, 3. lépés).

A horizont lejártakor automatikus, kézi beavatkozás nélkül. A becslés sora
**nem módosul**: a kimenetel külön sorban keletkezik, a becslés azonosítójára
hivatkozva. Ami egyszer lezárult, az nem íródik újra.

Kivezetett vagy felfüggesztett papírnál az utolsó ismert kereskedési nap
záróára zárja a becslést, `resolution_type` jelöléssel — így a kivezetés nem
tünteti el az eredményt (survivorship bias elleni védelem, spec/05, 2.5).

Futtatás:
    uv run python -m pipeline.resolve.run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import log as logging_setup
from pipeline.calendar import last_closed_session
from pipeline.config import HISTORY_START, RAW_BUCKET, load_settings
from pipeline.corporate import total_return_prices
from pipeline.features.run import _load_prices, _read_table, _write_table
from pipeline.forecast.run import FORECAST_PREFIX
from pipeline.ingest.partitions import ACTIONS_PATH, from_parquet
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.journal import resolve as journal_resolve
from pipeline.model.evaluate import apply_fdr, brier, compare, covered, hit, observations_needed, verdict

log = logging_setup.get_logger(__name__)

OUTCOMES_PREFIX = "outcomes"
LIVE_ARENA_PATH = "arena/live.parquet"
#: A piaci árazású baseline azonosítója (docs/piac-implikalt-baseline.md).
IMPLIED_ID = "market_implied"


def outcomes_path(year: int) -> str:
    return f"{OUTCOMES_PREFIX}/year={year}.parquet"


def load_forecasts(storage: Storage, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        for path in storage.list(RAW_BUCKET, f"{FORECAST_PREFIX}/{year}"):
            blob = storage.download(RAW_BUCKET, path)
            if blob is not None:
                frames.append(from_parquet(blob))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_outcomes(storage: Storage, years: list[int]) -> pd.DataFrame:
    frames = [f for f in (_read_table(storage, outcomes_path(y)) for y in years) if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _implied_scores(f: object, actual: float) -> dict[str, object]:
    """Az implikált baseline pontjai egy lezárt becslésre.

    Csak ott van értékük, ahol a becslés napján a lánc megbízható volt
    (`implied_status == "ok"`). A régi, implikált mezők nélküli csomagoknál
    és a „nem elérhető" napokon üresek — nem pótoljuk őket semmivel.
    """
    status = getattr(f, "implied_status", None)
    status = status if isinstance(status, str) else None
    empty = {
        "implied_status": status,
        "implied_prob": np.nan,
        "implied_hit": np.nan,
        "implied_brier": np.nan,
        "implied_covered": np.nan,
    }
    if status != "ok":
        return empty
    prob = float(getattr(f, "implied_prob", np.nan))
    low, high = float(getattr(f, "implied_band_low", np.nan)), float(getattr(f, "implied_band_high", np.nan))
    if not (np.isfinite(prob) and np.isfinite(low) and np.isfinite(high)):
        return empty
    y = np.array([actual])
    return {
        "implied_status": status,
        "implied_prob": prob,
        "implied_hit": float(hit(np.array([prob]), y)[0]),
        "implied_brier": float(brier(np.array([prob]), y)[0]),
        "implied_covered": float(covered(np.array([low]), np.array([high]), y)[0]),
    }


def resolve_due(
    forecasts: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    last_session: date,
    resolved_at: datetime,
) -> pd.DataFrame:
    """A lejárt, még le nem zárt becslések kiértékelése."""
    if forecasts.empty:
        return pd.DataFrame()
    tr = prices.assign(tr=total_return_prices(prices, actions))
    series = {
        instrument: g.set_index("date")["tr"].sort_index()
        for instrument, g in tr.groupby("instrument_id", sort=False)
    }

    rows = []
    due = forecasts[forecasts["target_session"].notna() & (forecasts["target_session"] <= last_session)]
    for f in due.itertuples():
        history = series.get(f.instrument_id)
        if history is None or f.session not in history.index:
            continue
        start = float(history.loc[f.session])
        if f.target_session in history.index:
            end, kind = float(history.loc[f.target_session]), "normal"
        else:
            # Nincs ár a célnapra: a papír kivezetésre került vagy fel van függesztve.
            earlier = history[history.index <= f.target_session]
            if earlier.empty or earlier.index[-1] <= f.session:
                continue
            end, kind = float(earlier.iloc[-1]), "delisted_or_halted"
        actual = float(np.log(end / start))
        rows.append(
            {
                "forecast_id": f.forecast_id,
                "instrument_id": f.instrument_id,
                "session": f.session,
                "target_session": f.target_session,
                "horizon": int(f.horizon),
                "regime": f.regime,
                "model_family": f.model_family,
                "model_version": f.model_version,
                "resolved_at": pd.Timestamp(resolved_at),
                "actual_return": actual,
                "prob_up": float(f.prob_up),
                "baseline_prob": float(f.baseline_prob),
                "hit": float(hit(np.array([f.prob_up]), np.array([actual]))[0]),
                "baseline_hit": float(hit(np.array([f.baseline_prob]), np.array([actual]))[0]),
                "brier": float(brier(np.array([f.prob_up]), np.array([actual]))[0]),
                "baseline_brier": float(brier(np.array([f.baseline_prob]), np.array([actual]))[0]),
                "covered": float(
                    covered(np.array([f.band_low]), np.array([f.band_high]), np.array([actual]))[0]
                ),
                "resolution_type": kind,
                **_implied_scores(f, actual),
            }
        )
    return pd.DataFrame(rows)


def _implied_comparisons(
    part: pd.DataFrame, meta: dict[str, object], horizon: int
) -> list[tuple[dict[str, object], object]]:
    """A modell a piaci árazás ellen — csak azokon a becsléseken, ahol az elérhető volt.

    Párosított összevetés: ugyanazok a becslések, ugyanazok a napok. A sáv
    lefedettségét nem a modellhez mérjük (ott nem a „több” a jobb), hanem a
    névleges 90%-hoz, a piaci árazás saját sorában.
    """
    if "implied_hit" not in part:
        return []
    usable = part[part["implied_hit"].notna()]
    if usable.empty:
        return []
    days = usable["session"].to_numpy()
    base = {
        **meta,
        "baseline_id": IMPLIED_ID,
        "first_observed": str(usable["session"].min()),
        "last_observed": str(usable["session"].max()),
    }
    out: list[tuple[dict[str, object], object]] = [
        (
            base,
            compare(
                "direction_accuracy",
                usable["hit"].to_numpy(),
                usable["implied_hit"].to_numpy(),
                days,
                horizon,
            ),
        ),
        (
            base,
            compare("brier", usable["brier"].to_numpy(), usable["implied_brier"].to_numpy(), days, horizon),
        ),
        (
            {**base, "subject_id": IMPLIED_ID, "baseline_id": "nominal_90"},
            compare(
                "coverage",
                usable["implied_covered"].to_numpy(),
                np.full(len(usable), 0.90),
                days,
                horizon,
            ),
        ),
    ]
    return out


def live_arena(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Élő mérési rekordok a lezárt becslésekből (spec/06, 8. fejezet)."""
    if outcomes.empty:
        return pd.DataFrame()
    records, comparisons = [], []
    for horizon in sorted(outcomes["horizon"].unique()):
        for regime in ["all", *sorted(outcomes["regime"].dropna().unique())]:
            part = outcomes[outcomes["horizon"] == horizon]
            if regime != "all":
                part = part[part["regime"] == regime]
            if part.empty:
                continue
            days = part["session"].to_numpy()
            meta = {
                "subject_id": f"{part['model_family'].iloc[0]} {part['model_version'].iloc[0]}",
                "scope": "universe",
                "horizon": int(horizon),
                "regime": regime,
                "baseline_id": "naive",
                "live": True,
                "first_observed": str(part["session"].min()),
                "last_observed": str(part["session"].max()),
            }
            for metric, model_rows, base_rows in (
                ("direction_accuracy", part["hit"].to_numpy(), part["baseline_hit"].to_numpy()),
                ("brier", part["brier"].to_numpy(), part["baseline_brier"].to_numpy()),
                (
                    "coverage",
                    part["covered"].to_numpy(),
                    np.full(len(part), 0.90),
                ),
            ):
                c = compare(metric, model_rows, base_rows, days, int(horizon))
                comparisons.append(
                    (meta if metric != "coverage" else {**meta, "baseline_id": "nominal_90"}, c)
                )
            comparisons.extend(_implied_comparisons(part, meta, int(horizon)))

    apply_fdr([c for _, c in comparisons])
    for meta, c in comparisons:
        is_rate = c.metric in {"direction_accuracy", "coverage"}
        records.append(
            {
                **meta,
                "metric": c.metric,
                "value": c.value,
                "baseline_value": c.baseline_value,
                "delta": c.delta,
                "n": c.n,
                "n_eff": c.n_eff,
                "hits": round(c.value * c.n) if is_rate else None,
                "baseline_hits": round(c.baseline_value * c.n) if is_rate else None,
                "p_value": c.p_value,
                "p_value_fdr": c.p_value_fdr,
                "n_tests": c.n_tests,
                "verdict": verdict(c),
                "observations_needed": observations_needed(c),
            }
        )
    return pd.DataFrame(records)


def resolve_journal(
    journal: journal_resolve.JournalStore,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    forecasts: pd.DataFrame,
    last: date,
) -> dict[str, int] | None:
    """A tézisek kiértékelése, a becslésekétől elszigetelve.

    Ha elhasal, a napló csak a hiba típusát írja (a részletben felhasználói
    adat lehetne), és a futás megy tovább: a tézisek hibája nem állíthatja meg
    a modell mérését.
    """
    try:
        tr = prices.assign(tr=total_return_prices(prices, actions))
        return journal_resolve.run(journal, journal_resolve.as_series(tr), forecasts, last)
    except Exception as exc:  # noqa: BLE001
        log.error("journal_resolve_failed", error=type(exc).__name__)
        return None


def run(
    storage: Storage,
    now: datetime,
    journal: journal_resolve.JournalStore | None = None,
) -> dict[str, object]:
    last = last_closed_session(now)
    years = list(range(2026, last.year + 1))
    forecasts = load_forecasts(storage, years)
    if forecasts.empty:
        log.info("resolve_nothing_to_do", reason="még nincs lementett becslés")
        return {"resolved_now": 0, "resolved_total": 0, "open": 0}

    known = load_outcomes(storage, years)
    done = set(known["forecast_id"]) if not known.empty else set()
    pending = forecasts[~forecasts["forecast_id"].isin(done)]

    prices = _load_prices(storage, list(range(max(HISTORY_START.year, last.year - 2), last.year + 1)))
    actions = _read_table(storage, ACTIONS_PATH)
    if actions is None:
        actions = pd.DataFrame(columns=["instrument_id", "date", "dividend", "split_ratio"])

    fresh = resolve_due(pending, prices, actions, last, now.astimezone(UTC))
    all_outcomes = pd.concat([known, fresh], ignore_index=True) if not fresh.empty else known

    if not fresh.empty:
        for year in sorted({d.year for d in all_outcomes["session"]}):
            part = all_outcomes[[d.year == year for d in all_outcomes["session"]]].reset_index(drop=True)
            _write_table(storage, outcomes_path(year), part)

    arena = live_arena(all_outcomes)
    if not arena.empty:
        _write_table(storage, LIVE_ARENA_PATH, arena)

    if journal is not None:
        resolve_journal(journal, prices, actions, forecasts, last)
        from pipeline.scenario.resolve import run_beside_journal

        tr = prices.assign(tr=total_return_prices(prices, actions))
        run_beside_journal(journal, storage, journal_resolve.as_series(tr), forecasts, last, "")

    # A modell-aréna (docs/modell-arena.md). Külön csomagból, külön táblába;
    # ha elbukik, a fő kiértékelés ettől nem marad el.
    arena_resolved = 0
    try:
        from pipeline.resolve import arena as model_arena

        arena_forecasts = model_arena.load_arena_forecasts(storage, years)
        if not arena_forecasts.empty:
            known_arena = model_arena.load_arena_outcomes(storage, years)
            done_arena = set(known_arena["forecast_id"]) if not known_arena.empty else set()
            fresh_arena = resolve_due(
                arena_forecasts[~arena_forecasts["forecast_id"].isin(done_arena)],
                prices,
                actions,
                last,
                now.astimezone(UTC),
            )
            all_arena = (
                pd.concat([known_arena, fresh_arena], ignore_index=True)
                if not fresh_arena.empty
                else known_arena
            )
            if not fresh_arena.empty:
                model_arena.write_arena_outcomes(storage, all_arena)
            if not all_arena.empty:
                model_arena.write_models_live(storage, model_arena.models_live(all_arena, all_outcomes))
            arena_resolved = len(all_arena)
    except Exception as error:  # noqa: BLE001 — a fő kiértékelés fontosabb
        log.warning("model_arena_failed", error=type(error).__name__)

    open_count = int(len(forecasts) - len(all_outcomes))
    summary: dict[str, object] = {
        "last_session": last.isoformat(),
        "resolved_now": len(fresh),
        "resolved_total": len(all_outcomes),
        "open": open_count,
        "live_records": len(arena),
        "arena_resolved": arena_resolved,
    }
    if not arena.empty:
        headline = arena[(arena["metric"] == "direction_accuracy") & (arena["regime"] == "all")]
        summary["live"] = [
            {
                "horizon": int(r.horizon),
                "n": int(r.n),
                "model": round(float(r.value), 4),
                "baseline": round(float(r.baseline_value), 4),
                "verdict": r.verdict,
            }
            for r in headline.itertuples()
        ]
    storage.upload(
        RAW_BUCKET,
        f"runs/resolve/{last.isoformat()}.json",
        json.dumps(summary, indent=2, default=str).encode(),
        "application/json",
    )
    log.info("resolve_done", **{k: v for k, v in summary.items() if k != "live"})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calyze kiértékelés")
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

    result = gated(storage, "resolve-equity", lambda: run(storage, datetime.now(UTC), journal))
    print(json.dumps(result if result is not None else {"status": "skipped"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
