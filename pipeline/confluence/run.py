"""A konfluencia-motor teljes historikus mérése (`docs/konfluencia.md`; spec/08, 3. fejezet).

A menet, pontosan a dokumentum szerint:

1. minden papíron a kiváltó események, és mindegyikhez a kontextus-maszk;
2. **felfedezés** (2018 végéig): minden kiváltó × kontextus-részhalmaz ×
   horizont, legalább 30 lezárt megfigyeléssel; Benjamini–Hochberg a teljes
   tesztelt családon;
3. **megerősítés** (2019-től): csak a felfedezésen átment kombinációk, saját
   baseline-nal, saját BH-korrekcióval.

A mérést ugyanaz a kód végzi, mint az arénákat (`compare`, `apply_fdr`,
`verdict`). A nyilvános naplóba csak darabszám kerül.

Futtatás:
    uv run python -m pipeline.confluence.run
    uv run python -m pipeline.confluence.run --local data --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import log as logging_setup
from pipeline.arena.evaluate import METRIC, baseline_direction, forward_outcomes
from pipeline.arena.signals import all_signals
from pipeline.config import HISTORY_START, RAW_BUCKET, load_settings
from pipeline.confluence.contexts import (
    CONTEXTS,
    combo_id,
    instrument_masks,
    is_trigger,
    mask_of,
    subsets,
)
from pipeline.features.run import REGIME_PATH, _load_prices, _read_table
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.model.config import CORE_HORIZONS
from pipeline.model.evaluate import MIN_OBSERVATIONS, Comparison, apply_fdr, compare, verdict
from pipeline.patterns.common import pivots
from pipeline.patterns.levels import walk_levels
from pipeline.patterns.run import instrument_events
from pipeline.publish.run import DISPLAY_BUCKET
from pipeline.universe import active_on, load_universe

log = logging_setup.get_logger(__name__)

# A konfluencia tesztelt családja a mérés előtt rögzült (2026-09-27): az 1 és
# a 120 napos időtáv bevétele új keresés lenne, ezért itt csak a régi három fut.
HORIZONS = CORE_HORIZONS

#: A két időszak határa (3. fejezet). Egyetlen érték, nem mozog.
DISCOVERY_END = date(2018, 12, 31)
CONFIRMATION_START = date(2019, 1, 1)

RESULTS_PATH = "confluence/results.parquet"
DISPLAY_FILE = "confluence.json"
#: Szétbontva (`docs/terv-egress.md`, J4): a teljes fájl 1,4 MB volt, és a
#: Bizonyíték-tábla minden látogatáskor letöltötte, pedig csak a megerősítő
#: időszak sorai kellenek neki. A Konfluencia lap horizontonként kér.
BOARD_FILE = "confluence/board.json"
HORIZON_FILE = "confluence/h{horizon}.json"

EVENT_COLUMNS = ["instrument_id", "date", "trigger", "direction", "mask"]


# ---------------------------------------------------------------- események


def instrument_triggers(
    instrument: str,
    frame: pd.DataFrame,
    indicator_rows: pd.DataFrame,
    regime: Mapping[date, str],
) -> pd.DataFrame:
    """Egy papír kiváltó eseményei, a kiváltó irányához tartozó kontextus-maszkkal."""
    data = frame.sort_values("date").reset_index(drop=True)
    walk = walk_levels(data)
    found = pivots(data)
    long_mask, short_mask = instrument_masks(data, walk, found, regime)
    position = {d: i for i, d in enumerate(data["date"])}

    rows: list[tuple[object, str, str]] = [
        (r["date"], str(r["rule"]), str(r["direction"]))
        for r in instrument_events(instrument, data, walk)
        if is_trigger(str(r["rule"]))
    ]
    rows += [
        (d, str(rule), str(direction))
        for d, rule, direction in indicator_rows[["date", "rule", "direction"]].itertuples(index=False)
    ]

    out = []
    for day, trigger, direction in rows:
        i = position.get(day)
        if i is None:
            continue
        mask = long_mask[i] if direction == "long" else short_mask[i]
        out.append((instrument, day, trigger, direction, int(mask)))
    return pd.DataFrame(out, columns=EVENT_COLUMNS)


def _triggers_of(item: tuple[str, pd.DataFrame, pd.DataFrame, dict[date, str]]) -> pd.DataFrame:
    return instrument_triggers(*item)


def all_triggers(
    prices: pd.DataFrame, regime: Mapping[date, str], workers: int | None = None
) -> pd.DataFrame:
    """Az egész univerzum kiváltói. A papírok függetlenek, ezért párhuzamosan."""
    indicators = all_signals(prices)
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    by_instrument = {str(k): g for k, g in indicators.groupby("instrument_id")}
    empty = indicators.iloc[0:0]
    regime_dict = dict(regime)
    items = [
        (str(i), g, by_instrument.get(str(i), empty), regime_dict)
        for i, g in prices.groupby("instrument_id", sort=True)
    ]
    if workers == 1 or len(items) < 8:
        parts = [_triggers_of(item) for item in items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_triggers_of, items, chunksize=4))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EVENT_COLUMNS)


# ---------------------------------------------------------------- mérés


@dataclass(frozen=True)
class Sample:
    """Egy időszak egy horizontja: a kiváltók találatai a baseline mellett."""

    trigger: np.ndarray
    mask: np.ndarray
    hit: np.ndarray
    baseline_hit: np.ndarray
    day: np.ndarray


def period_sample(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int,
    start: date | None,
    end: date | None,
) -> Sample:
    """A kiváltók kimenetele egy időszakon belül.

    A kimenetel nem nyúlhat át az időszak végén: az árakat is az időszak
    végéig vágjuk, így csak a határon belül lezárult horizont számít. A
    baseline az időszak saját sodródása.
    """
    px = prices if end is None else prices[prices["date"] <= end]
    outcomes = forward_outcomes(px, horizon)
    if start is not None:
        outcomes = outcomes[outcomes["date"] >= start]
    base_up = baseline_direction(outcomes)
    joined = events.merge(outcomes, on=["instrument_id", "date"], how="inner")
    went_up = joined["up"].to_numpy() > 0.5
    wants_up = joined["direction"].to_numpy() == "long"
    base = joined["instrument_id"].map(base_up).fillna(False).to_numpy(dtype=bool)
    return Sample(
        trigger=joined["trigger"].to_numpy(),
        mask=joined["mask"].to_numpy(dtype=np.int64),
        hit=(went_up == wants_up).astype("float64"),
        baseline_hit=(went_up == base).astype("float64"),
        day=pd.to_datetime(joined["date"]).astype("int64").to_numpy(),
    )


@dataclass(frozen=True)
class Task:
    trigger: str
    horizon: int
    combos: tuple[tuple[str, ...], ...]
    mask: np.ndarray
    hit: np.ndarray
    baseline_hit: np.ndarray
    day: np.ndarray


def _run_task(task: Task) -> list[tuple[str, int, int, Comparison | None]]:
    """(kombináció, horizont, n, összevetés) — 30 alatt összevetés nélkül (nincs p-érték)."""
    out: list[tuple[str, int, int, Comparison | None]] = []
    for contexts in task.combos:
        need = mask_of(contexts)
        chosen = (task.mask & need) == need
        n = int(chosen.sum())
        if n < MIN_OBSERVATIONS:
            out.append((combo_id(task.trigger, contexts), task.horizon, n, None))
            continue
        comparison = compare(
            METRIC, task.hit[chosen], task.baseline_hit[chosen], task.day[chosen], task.horizon
        )
        out.append((combo_id(task.trigger, contexts), task.horizon, n, comparison))
    return out


def _tasks(sample: Sample, horizon: int, wanted: Mapping[str, list[tuple[str, ...]]]) -> list[Task]:
    tasks = []
    for trigger, combos in wanted.items():
        rows = sample.trigger == trigger
        if not rows.any() or not combos:
            continue
        tasks.append(
            Task(
                trigger,
                horizon,
                tuple(combos),
                sample.mask[rows],
                sample.hit[rows],
                sample.baseline_hit[rows],
                sample.day[rows],
            )
        )
    return tasks


def _execute(tasks: list[Task], workers: int | None) -> list[tuple[str, int, int, Comparison | None]]:
    if workers == 1 or len(tasks) < 4:
        return [r for t in tasks for r in _run_task(t)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [r for part in pool.map(_run_task, tasks, chunksize=1) for r in part]


def measure(events: pd.DataFrame, prices: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """A teljes konfluencia-mérés: felfedezés, BH, megerősítés, BH."""
    triggers = sorted(events["trigger"].unique())
    family = {t: subsets(t) for t in triggers}
    direction = events.drop_duplicates("trigger").set_index("trigger")["direction"].to_dict()

    # 1. felfedezés
    discovery: list[tuple[str, int, int, Comparison | None]] = []
    for horizon in HORIZONS:
        sample = period_sample(events[events["date"] <= DISCOVERY_END], prices, horizon, None, DISCOVERY_END)
        discovery += _execute(_tasks(sample, horizon, family), workers)
    tested = [c for _, _, _, c in discovery if c is not None]
    apply_fdr(tested)

    rows: dict[tuple[str, int], dict[str, object]] = {}
    passed: dict[int, dict[str, list[tuple[str, ...]]]] = {h: {} for h in HORIZONS}
    for combo, horizon, n, comparison in discovery:
        trigger, *contexts = combo.split("+")
        row: dict[str, object] = {
            "combo": combo,
            "trigger": trigger,
            "contexts": "+".join(contexts),
            "direction": direction[trigger],
            "horizon": horizon,
            **_stats("disc", n, comparison),
        }
        if comparison is None:
            row["status"] = "too_early"
        elif row["disc_verdict"] == "better_significant":
            row["status"] = "found"
            passed[horizon].setdefault(trigger, []).append(tuple(contexts))
        else:
            row["status"] = "not_found"
        rows[(combo, horizon)] = row

    # 2. megerősítés — csak a felfedezésen átment kombinációk
    confirmation: list[tuple[str, int, int, Comparison | None]] = []
    for horizon in HORIZONS:
        if not passed[horizon]:
            continue
        sample = period_sample(
            events[events["date"] >= CONFIRMATION_START], prices, horizon, CONFIRMATION_START, None
        )
        confirmation += _execute(_tasks(sample, horizon, passed[horizon]), workers)
    apply_fdr([c for _, _, _, c in confirmation if c is not None])
    for combo, horizon, n, comparison in confirmation:
        row = rows[(combo, horizon)]
        row.update(_stats("conf", n, comparison))
        if comparison is None:
            row["status"] = "found_too_early"
        elif row["conf_verdict"] == "better_significant":
            row["status"] = "confirmed"
        else:
            row["status"] = "found_not_confirmed"

    table = pd.DataFrame(list(rows.values()))
    for column in (
        "conf_n",
        "conf_n_eff",
        "conf_hit",
        "conf_baseline",
        "conf_delta",
        "conf_p",
        "conf_verdict",
    ):
        if column not in table:
            table[column] = None
    return table.sort_values(["horizon", "trigger", "combo"]).reset_index(drop=True)


def _stats(prefix: str, n: int, c: Comparison | None) -> dict[str, object]:
    if c is None:
        return {f"{prefix}_n": n}
    return {
        f"{prefix}_n": n,
        f"{prefix}_n_eff": round(float(c.n_eff), 1),
        f"{prefix}_hit": round(float(c.value), 4),
        f"{prefix}_baseline": round(float(c.baseline_value), 4),
        f"{prefix}_delta": round(float(c.delta), 4),
        f"{prefix}_p": round(float(c.p_value_fdr), 4),
        f"{prefix}_verdict": verdict(c),
    }


# ---------------------------------------------------------------- kimenet


def summary_of(table: pd.DataFrame, events: pd.DataFrame) -> dict[str, int]:
    """Csak darabszámok: a nyilvános naplóba is ez kerül."""
    status = table["status"] if not table.empty else pd.Series(dtype=str)
    return {
        "events": len(events),
        "triggers": int(events["trigger"].nunique()) if not events.empty else 0,
        "defined": len(table),
        "tested": int((status != "too_early").sum()),
        "found": int(status.isin(["found", "found_not_confirmed", "found_too_early", "confirmed"]).sum()),
        "confirmed": int((status == "confirmed").sum()),
    }


def display_payload(table: pd.DataFrame, events: pd.DataFrame, now: datetime) -> dict[str, object]:
    """A felületnek: az összesítés és minden TESZTELT kombináció (a 30 alattiak csak darabszámként)."""
    dates = pd.to_datetime(events["date"]) if not events.empty else pd.Series(dtype="datetime64[ns]")
    tested = table[table["status"] != "too_early"] if not table.empty else table
    keys = ["n", "n_eff", "hit", "baseline", "delta", "p", "verdict"]
    rows = []
    for r in tested.to_dict("records"):
        row: dict[str, object] = {
            "combo": r["combo"],
            "trigger": r["trigger"],
            "contexts": [c for c in str(r["contexts"]).split("+") if c],
            "direction": r["direction"],
            "horizon": int(r["horizon"]),
            "status": r["status"],
            "discovery": {k: _plain(r.get(f"disc_{k}")) for k in keys},
        }
        if r["status"] in {"confirmed", "found_not_confirmed", "found_too_early"}:
            row["confirmation"] = {k: _plain(r.get(f"conf_{k}")) for k in keys}
        rows.append(row)
    return {
        "generated_at": now.astimezone(UTC).replace(microsecond=0).isoformat(),
        "horizons": list(HORIZONS),
        "contexts": list(CONTEXTS),
        "discovery": {"from": None if dates.empty else str(dates.min().date()), "to": str(DISCOVERY_END)},
        "confirmation": {
            "from": str(CONFIRMATION_START),
            "to": None if dates.empty else str(dates.max().date()),
        },
        **summary_of(table, events),
        "survivorship_bias": True,
        "rows": rows,
    }


def split_payload(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Útvonal → csomag: a tábla szelete és horizontonként egy fájl."""
    rows = payload.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    head = {k: v for k, v in payload.items() if k != "rows"}
    out: dict[str, dict[str, object]] = {
        BOARD_FILE: {
            "generated_at": payload.get("generated_at"),
            "confirmation": payload.get("confirmation"),
            "rows": [
                {k: r[k] for k in ("combo", "trigger", "contexts", "horizon", "confirmation")}
                for r in rows
                if "confirmation" in r
            ],
        }
    }
    for horizon in payload.get("horizons", []):  # type: ignore[union-attr]
        out[HORIZON_FILE.format(horizon=horizon)] = {
            **head,
            "rows": [r for r in rows if r["horizon"] == horizon],
        }
    return out


def _plain(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(storage: Storage, now: datetime, dry_run: bool = False) -> dict[str, object]:
    prices = _load_prices(storage, list(range(HISTORY_START.year, now.year + 1)))
    if prices.empty:
        raise RuntimeError("Nincs árfolyam a tárban — előbb az ingest fusson le.")
    universe = active_on(load_universe(), now.astimezone(UTC).date())
    prices = prices[prices["instrument_id"].isin(set(universe["instrument_id"]))].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    regime_table = _read_table(storage, REGIME_PATH)
    regime: dict[date, str] = {}
    if regime_table is not None and not regime_table.empty:
        labels = regime_table.dropna(subset=["regime"])
        regime = dict(zip(pd.to_datetime(labels["date"]).dt.date, labels["regime"].astype(str), strict=True))

    log.info("confluence_start", instruments=int(prices["instrument_id"].nunique()), rows=len(prices))
    events = all_triggers(prices, regime)
    log.info("confluence_events", events=len(events), triggers=int(events["trigger"].nunique()))
    table = measure(events, prices)
    summary = summary_of(table, events)

    if dry_run:
        log.info("confluence_dry_run", **summary)
        return {**summary, "status": "dry_run"}

    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False, compression="zstd")
    storage.upload(RAW_BUCKET, RESULTS_PATH, buffer.getvalue(), "application/octet-stream")
    for path, payload in split_payload(display_payload(table, events, now)).items():
        storage.upload(
            DISPLAY_BUCKET,
            path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            "application/json",
        )
    log.info("confluence_done", **summary)
    return {**summary, "status": "saved"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Konfluencia-motor, teljes historikus mérés")
    parser.add_argument("--local", type=Path, default=None, help="helyi tár a Supabase helyett")
    parser.add_argument("--dry-run", action="store_true", help="számol, de nem ír")
    args = parser.parse_args()
    logging_setup.configure()
    storage: Storage
    if args.local is not None:
        storage = LocalStorage(args.local)
    else:
        settings = load_settings()
        storage = SupabaseStorage(settings.supabase_url or "", settings.supabase_secret_key or "")
    result = run(storage, datetime.now(UTC), dry_run=args.dry_run)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
