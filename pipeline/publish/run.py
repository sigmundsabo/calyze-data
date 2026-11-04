"""Megjelenítésre kész JSON-ok a webappnak (spec/03; spec/05, 1. fejezet).

A becslések parquetben, a privát tárban élnek; a felület nem lát oda, és nem
is olvashat parquetet. Ez a lépés a napi futás után átfordítja őket abba a
formába, amit a képernyők használnak, és egy **nem publikus** tárolóba
tölti fel, ahonnan csak belépett felhasználó tölthet le.

Ez tartja be az adatlicencet is: az ingyenes források adatát csak regisztrált
felhasználó láthatja, regisztrálni pedig egyelőre csak meghívókóddal lehet
(`docs/adatlicencek.md` a másik repóban).

Futtatás:
    uv run python -m pipeline.publish.run
    uv run python -m pipeline.publish.run --local data --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import log as logging_setup
from pipeline.calendar import last_closed_session, session_lag
from pipeline.config import RAW_BUCKET, load_settings
from pipeline.features.run import REGIME_PATH, _load_prices, _read_table
from pipeline.forecast.run import FORECAST_PREFIX
from pipeline.ingest.partitions import from_parquet
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.model.config import HORIZONS, MIN_HISTORY_SESSIONS
from pipeline.model.evaluate import MIN_OBSERVATIONS
from pipeline.publish.postmortem import build_postmortems
from pipeline.resolve.run import IMPLIED_ID, LIVE_ARENA_PATH, load_outcomes
from pipeline.universe import active_on, load_universe

log = logging_setup.get_logger(__name__)

#: A megjelenítésre szánt tároló. Nem publikus: belépés nélkül nem olvasható.
DISPLAY_BUCKET = "display"

#: Ennyi kereskedési nap ára kerül a charthoz (spec/03, 2.1: 90–250 nap).
HISTORY_SESSIONS = 250

#: A chart-munkaasztal idősora (spec/03, 2.2): öt év látható ablak, plusz 200
#: nap bemelegítés, hogy a 200 napos átlag az ablak első napján is létezzen.
#: Külön fájlban él, hogy a papír-nézet a saját 250 napjával gyors maradjon.
WORKBENCH_SESSIONS = 5 * 252 + 200
#: Ennyi naptári évet kell betölteni a munkaasztal idősorához.
WORKBENCH_YEARS = 7
#: A részvényes megjelenítés feltöltési manifesztje (útvonal → sha256).
MANIFEST_PATH = ".manifest.json"

#: Ennyi napra visszamenőleg mutatjuk, hogyan változott a becslés.
TIMELINE_SESSIONS = 90


def _num(value: object, digits: int = 6) -> float | None:
    """Szám JSON-ba: a NaN nem szám, és nem is nulla — hiányzó értékként megy ki."""
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else round(out, digits)


def _dumps(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")


def load_forecast_history(storage: Storage, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        for path in storage.list(RAW_BUCKET, f"{FORECAST_PREFIX}/{year}"):
            blob = storage.download(RAW_BUCKET, path)
            if blob is not None:
                frames.append(from_parquet(blob))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["session", "instrument_id", "horizon"]).reset_index(drop=True)


def next_resolutions(forecasts: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, str | None]:
    """Horizontonként mikor zárul a legközelebbi még nyitott becslés.

    Ez adja a felület „mikorra várható” mondatát: mintaszám nélkül a dátum az
    egyetlen őszinte állítás, amit tenni tudunk.
    """
    done = set(outcomes["forecast_id"]) if not outcomes.empty else set()
    out: dict[str, str | None] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        if forecasts.empty:
            out[key] = None
            continue
        part = forecasts[(forecasts["horizon"] == horizon) & (~forecasts["forecast_id"].isin(done))]
        targets = [t for t in part["target_session"].dropna().tolist()]
        out[key] = str(min(targets)) if targets else None
    return out


#: Az őszinteség-kapu kérdéseinek száma és a mutatott múlt hossza.
HONESTY_QUESTIONS = 12
HONESTY_WINDOW = 120
HONESTY_HORIZON = 20
# A bizonytalansági sáv határai az ablak saját szórásában mérve: alatta zaj,
# fölötte sokk. Egyik sem mond semmit a magabiztosságról.
HONESTY_MIN_Z = 0.25
HONESTY_MAX_Z = 1.5


def build_honesty(
    prices: pd.DataFrame, session: date, count: int = HONESTY_QUESTIONS
) -> list[dict[str, object]]:
    """Kérdések rejtett múltbeli chartokból (spec/02, F7).

    A papír neve és a dátum szándékosan hiányzik: ha felismerhető lenne,
    a kapu nem a magabiztosságot mérné, hanem az emlékezetet. A sorozat
    100-ra normálva megy ki, így a szintből sem lehet visszakövetkeztetni.

    A kivágási pont nem lehet akármelyik nap. Két szűrőn megy át:

    1. **A kimenetel legyen valóban bizonytalan.** A jövőbeli elmozdulást a
       saját ablak napi szórásához mérjük (`z`). Ami alatta van a
       `HONESTY_MIN_Z`-nek, az zajról szól — ott a fel/le kérdés érme, de
       nem azért, mert nehéz, hanem mert nincs mit eltalálni. Ami a
       `HONESTY_MAX_Z` fölött van, az sokk: azt utólag nézve nyilvánvaló, és
       a magabiztosságról nem mond semmit.
    2. **A kérdéskészlet legyen kiegyensúlyozott.** Fele emelkedő, fele
       csökkenő kimenetel. Enélkül a kapu egy „mindig felfelé" válasszal
       megnyerhető lenne, és pont az ellenkezőjét tanítaná annak, amiért van.

    Egy papírról legfeljebb egy kérdés kerül be, hogy a készlet ne egyetlen
    részvény történetét kérdezze vissza ötször.

    A választás a session dátumából magolt véletlennel történik: ugyanarra a
    napra ugyanazok a kérdések, de naponta mások.
    """
    if prices.empty:
        return []

    rng = np.random.default_rng(int(session.strftime("%Y%m%d")))
    frame = prices.sort_values(["instrument_id", "date"])
    ids = frame["instrument_id"].dropna().unique()
    if len(ids) == 0:
        return []

    # Fele-fele, páratlan kérdésszámnál az emelkedő kap eggyel többet.
    want = {True: (count + 1) // 2, False: count // 2}
    picked: dict[bool, list[dict[str, object]]] = {True: [], False: []}
    used: set[str] = set()

    # Első kör: minden kérdés más papírról. Ha így nem jön össze a készlet
    # (kicsi univerzum), a második kör megengedi az ismétlést — inkább legyen
    # kérdés, mint üres kapu.
    for distinct in (True, False):
        attempts = 0
        while sum(len(v) for v in picked.values()) < count and attempts < count * 80:
            attempts += 1
            questions_step(rng, ids, frame, picked, want, used, distinct)
    # Keverés, hogy a fel/le ne váltakozzon felismerhető mintában.
    questions = picked[True] + picked[False]
    order = rng.permutation(len(questions))
    return [{"id": f"q{i + 1}", **questions[int(j)]} for i, j in enumerate(order)]


def questions_step(
    rng: np.random.Generator,
    ids: np.ndarray,
    frame: pd.DataFrame,
    picked: dict[bool, list[dict[str, object]]],
    want: dict[bool, int],
    used: set[str],
    distinct: bool,
) -> None:
    """Egy próbálkozás: kiválaszt egy papírt és egy kivágási pontot.

    Ha a kérdés nem felel meg valamelyik feltételnek, nem történik semmi —
    a hívó újra próbálja.
    """
    instrument = str(ids[rng.integers(0, len(ids))])
    if distinct and instrument in used:
        return
    part = frame[frame["instrument_id"] == instrument].reset_index(drop=True)
    needed = HONESTY_WINDOW + HONESTY_HORIZON
    if len(part) < needed + 1:
        return
    cut = int(rng.integers(HONESTY_WINDOW, len(part) - HONESTY_HORIZON))
    window = part.iloc[cut - HONESTY_WINDOW : cut]["close"].astype("float64").to_numpy()
    future = float(part.iloc[cut + HONESTY_HORIZON - 1]["close"])
    last = float(window[-1])
    if not np.isfinite(last) or last <= 0 or not np.isfinite(future) or np.isnan(window).any():
        return

    ret = future / last - 1.0
    z = _uncertainty(window, ret)
    if z is None or z < HONESTY_MIN_Z or z > HONESTY_MAX_Z:
        return

    up = bool(future > last)
    if len(picked[up]) >= want[up]:
        return

    used.add(instrument)
    picked[up].append(
        {
            "series": [round(float(v) / last * 100.0, 3) for v in window],
            "horizon": HONESTY_HORIZON,
            "outcome_up": up,
            "outcome_return": round(ret, 6),
        }
    )


def _uncertainty(window: np.ndarray, forward_return: float) -> float | None:
    """A jövőbeli elmozdulás az ablak saját szórásában mérve.

    Nem abszolút százalék, mert egy 3%-os mozgás egy csendes ETF-nél nagy,
    egy ingadozó papírnál semmi. `None`, ha az ablakból nem számolható
    szórás — akkor a kérdés kimarad.
    """
    daily = np.diff(np.log(window))
    sigma = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return abs(forward_return) / (sigma * math.sqrt(HONESTY_HORIZON))


def build_arena(arena: pd.DataFrame) -> list[dict[str, object]]:
    """Az élő mérési rekordok. Ami 30 megfigyelés alatt van, az „too early”."""
    if arena.empty:
        return []
    records = []
    for row in arena.to_dict("records"):
        records.append(
            {
                "subject": row.get("subject_id"),
                "scope": row.get("scope"),
                "horizon": int(row["horizon"]),
                "regime": row.get("regime"),
                "metric": row.get("metric"),
                "baseline": row.get("baseline_id"),
                "value": _num(row.get("value")),
                "baseline_value": _num(row.get("baseline_value")),
                "delta": _num(row.get("delta")),
                "n": int(row.get("n") or 0),
                "n_eff": _num(row.get("n_eff"), 2),
                "p_value": _num(row.get("p_value")),
                "p_value_fdr": _num(row.get("p_value_fdr")),
                "n_tests": int(row.get("n_tests") or 0),
                "verdict": row.get("verdict"),
                "observations_needed": int(row["observations_needed"])
                if pd.notna(row.get("observations_needed"))
                else None,
                "first_observed": row.get("first_observed"),
                "last_observed": row.get("last_observed"),
            }
        )
    return records


def headline(records: list[dict[str, object]]) -> dict[str, object] | None:
    """A kumulált állás egy sorban: az irány-pontosság 20 napon, minden rezsimben.

    A verdict a baseline-család **legkeményebb** tagja ellen szól
    (docs/piac-implikalt-baseline.md, 6.): amelyik ellen a modell a
    legrosszabbul áll. Tag csak az lehet, amelyiknek már van elég
    megfigyelése — a piaci árazás addig nem „győzhet” és nem is „veszíthet”,
    amíg 30 párosított becslés sincs mögötte. A `family` mindig kiírja, kik a
    tagok, hogy a felület megmondhassa, mihez mérünk.
    """
    candidates = [
        r
        for r in records
        if r["metric"] == "direction_accuracy" and r["horizon"] == 20 and r["regime"] == "all"
    ]
    if not candidates:
        return records[0] if records else None
    measured = [r for r in candidates if int(r["n"] or 0) >= MIN_OBSERVATIONS]  # type: ignore[call-overload]
    naive = [r for r in candidates if r["baseline"] != IMPLIED_ID]
    pool = measured or naive or candidates
    hardest = min(pool, key=lambda r: r["delta"] if r["delta"] is not None else math.inf)  # type: ignore[arg-type,return-value]
    family = [
        {"baseline": r["baseline"], "n": r["n"], "counted": any(r is m for m in pool)}
        for r in sorted(candidates, key=lambda r: str(r["baseline"]))
    ]
    return {**hardest, "family": family}


def build_latest(
    session: date,
    universe: pd.DataFrame,
    today_forecasts: pd.DataFrame,
    all_forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
    arena_records: list[dict[str, object]],
    regime: pd.DataFrame | None,
    now: datetime,
) -> dict[str, object]:
    resolved_recently = pd.DataFrame()
    if not outcomes.empty and "resolved_at" in outcomes:
        last_day = pd.to_datetime(outcomes["resolved_at"]).max()
        if pd.notna(last_day):
            resolved_recently = outcomes[pd.to_datetime(outcomes["resolved_at"]) == last_day]

    regime_label, regime_value = None, None
    if regime is not None and not regime.empty:
        row = regime[regime["date"] == session]
        if not row.empty:
            regime_label = str(row["regime"].iloc[0])
            # A „stress” a rezsim folytonos mérőszáma (0–1): a felület ezt
            # mutatja a címke mellett, hogy a határ közelsége is látszódjon.
            regime_value = _num(row["stress"].iloc[0]) if "stress" in row else None

    return {
        "session": str(session),
        "generated_at": now.astimezone(UTC).replace(microsecond=0).isoformat(),
        # Hány kereskedési nappal marad el az adat a mai naptól. Nulla a
        # rendes eset. Ha nem nulla, a felület kimondja — a felhasználó ne
        # abból jöjjön rá, hogy a dátum ismerősnek tűnik.
        "source_lag_sessions": session_lag(session, last_closed_session(now)),
        "regime": regime_label,
        "regime_stress": regime_value,
        "universe": len(universe),
        "instruments_with_forecast": int(today_forecasts["instrument_id"].nunique())
        if not today_forecasts.empty
        else 0,
        "forecasts_total": len(all_forecasts),
        "forecasts_open": int(len(all_forecasts) - len(outcomes)),
        "resolved": {
            "on": str(pd.to_datetime(resolved_recently["resolved_at"]).max().date())
            if not resolved_recently.empty
            else None,
            "count": len(resolved_recently),
            "hits": int(resolved_recently["hit"].sum()) if not resolved_recently.empty else 0,
            "baseline_hits": int(resolved_recently["baseline_hit"].sum())
            if not resolved_recently.empty
            else 0,
        },
        "verdict": headline(arena_records),
        "next_resolution": next_resolutions(all_forecasts, outcomes),
        "min_observations": MIN_OBSERVATIONS,
        "upcoming_events": _upcoming(session),
        "market_shock": _market_shock(today_forecasts),
    }


def _market_shock(today: pd.DataFrame) -> dict[str, object]:
    """A piaci szintű jelzés a mai becslés-csomagból (docs/sokk-detektor.md)."""
    if today.empty or "shock_status" not in today:
        return {"status": "not_collected"}
    row = today.iloc[0]
    if row["shock_status"] != "ok":
        return {"status": str(row["shock_status"])}
    market = [x for x in str(row.get("shock_market") or "").split(",") if x]
    flagged = int((today.drop_duplicates("instrument_id")["shock_signals"].fillna("") != "").sum())
    return {
        "status": "ok",
        "signals": market,
        "withheld": bool(row.get("withheld", False)),
        "instruments_flagged": flagged,
    }


def _news(session: date, latest: dict[str, object]) -> dict[str, object]:
    from pipeline.news.official import official_news

    shock = latest.get("market_shock")
    signals = (
        list(shock.get("signals", [])) if isinstance(shock, dict) and shock.get("status") == "ok" else []
    )
    try:
        return official_news(session, signals)
    except Exception as error:  # noqa: BLE001
        log.warning("news_failed", error=type(error).__name__)
        return {"status": "failed"}


def _upcoming(session: date) -> list[dict[str, str]]:
    from pipeline.events.calendar import upcoming

    return [
        {"id": e.id, "kind": e.kind, "day": e.day.isoformat(), "time_et": e.time_et}
        for e in upcoming(session)
    ]


def build_index(
    universe: pd.DataFrame, today_forecasts: pd.DataFrame, history: pd.DataFrame
) -> list[dict[str, object]]:
    """A kereséshez: minden papír, adatminőség-jelölővel (spec/02, F1, 2. lépés)."""
    covered = set(today_forecasts["instrument_id"]) if not today_forecasts.empty else set()
    counts = history.groupby("instrument_id")["date"].count().to_dict() if not history.empty else {}
    rows = []
    for row in universe.to_dict("records"):
        sessions = int(counts.get(row["instrument_id"], 0))
        rows.append(
            {
                "id": row["instrument_id"],
                "ticker": row["ticker"],
                "name": row["name"],
                "asset_class": row.get("asset_class"),
                "sector": row.get("sector"),
                "sessions": sessions,
                "forecastable": row["instrument_id"] in covered,
                # Ha nincs elég múlt, a felület halványítja a sort, és megmondja, miért.
                "reason": None
                if row["instrument_id"] in covered
                else ("short_history" if sessions < MIN_HISTORY_SESSIONS else "no_forecast"),
            }
        )
    return sorted(rows, key=lambda r: str(r["ticker"]))


def format_forecast(row: dict[str, object]) -> dict[str, object]:
    """Egy lementett becslés-sor a felület formájában.

    Az élő papír-nézet és az időgép UGYANEZT a függvényt használja: a múltbeli
    nap nem egy másik képletből áll össze, hanem pontosan ugyanabból, amiből
    aznap is összeállt volna.
    """
    raw = row.get("contributions")
    try:
        contributions = json.loads(raw) if isinstance(raw, str) and raw else []
    except json.JSONDecodeError:
        contributions = []
    target = row.get("target_session")
    # Visszatartott becslés: a szám a lenyomat alatt megvan, de a felületre
    # nem kerül ki — az időgép is úgy mutatja, ahogy aznap látszott.
    if row.get("withheld") is True or row.get("withheld") == 1:
        return {
            "horizon": int(row["horizon"]),  # type: ignore[arg-type]
            "target_session": str(target) if target is not None and pd.notna(target) else None,
            "withheld": True,
            "prob_up": None,
            "baseline_prob": None,
            "baseline": row.get("baseline_id"),
            "expected_return": None,
            "band_low": None,
            "band_high": None,
            "scale": None,
            "price_low": None,
            "price_high": None,
            "expected_price": None,
            "made_at": str(row.get("made_at")),
            "contributions": [],
            "implied": {"status": "not_collected"},
            "calendar": _calendar(row),
            "shock": _shock(row),
        }
    return {
        "horizon": int(row["horizon"]),  # type: ignore[arg-type]
        "target_session": str(target) if target is not None and pd.notna(target) else None,
        "prob_up": _num(row.get("prob_up")),
        "baseline_prob": _num(row.get("baseline_prob")),
        "baseline": row.get("baseline_id"),
        "expected_return": _num(row.get("expected_return")),
        "band_low": _num(row.get("band_low")),
        "band_high": _num(row.get("band_high")),
        # a Forgatókönyv nevezője; a régi (2026-10-20 előtti) csomagokban nincs
        "scale": _num(row.get("scale"), 6),
        "price_low": _num(row.get("price_low"), 4),
        "price_high": _num(row.get("price_high"), 4),
        "expected_price": _num(row.get("expected_price"), 4),
        "made_at": str(row.get("made_at")),
        "contributions": contributions,
        "implied": _implied(row),
        "calendar": _calendar(row),
        "shock": _shock(row),
        "withheld": False,
    }


def _shock(row: dict[str, object]) -> dict[str, object]:
    """A sokk-detektor jelei a becslés napján (docs/sokk-detektor.md)."""
    status = row.get("shock_status")
    if not isinstance(status, str):
        return {"status": "not_collected"}
    if status != "ok":
        return {"status": status}

    def _split(value: object) -> list[str]:
        return [x for x in str(value).split(",") if x] if isinstance(value, str) else []

    def _text(value: object) -> str | None:
        return value if isinstance(value, str) and value else None

    return {
        "status": "ok",
        "signals": _split(row.get("shock_signals")),
        "market": _split(row.get("shock_market")),
        "scope": _text(row.get("shock_scope")),
        "persistence": _text(row.get("shock_persistence")),
        "tractability": _text(row.get("shock_tractability")),
    }


def _calendar(row: dict[str, object]) -> dict[str, object]:
    """Az ablakba eső ütemezett események (docs/naptar.md).

    A régi csomagokban nincs ilyen oszlop: ott `not_collected`, hogy a felület
    ne állítsa, hogy az ablakban nem volt esemény, amikor nem is figyeltük.
    """
    from pipeline.events.calendar import EVENTS, uncovered_kinds

    flag = row.get("calendar_flag")
    if not isinstance(flag, str):
        return {"status": "not_collected"}
    by_id = {e.id: e for e in EVENTS}
    events = [by_id[i] for i in flag.split(",") if i in by_id]
    target = row.get("target_session")
    end = date.fromisoformat(str(target)) if target is not None and pd.notna(target) else None
    return {
        "status": "ok",
        "events": [
            {"id": e.id, "kind": e.kind, "day": e.day.isoformat(), "time_et": e.time_et} for e in events
        ],
        # a fajták, amelyekről a célnapig még nincs teljes naptárunk
        "uncovered": uncovered_kinds(end) if end is not None else [],
    }


def _implied(row: dict[str, object]) -> dict[str, object]:
    """A piaci árazás a becslés napján — vagy az ok, amiért nincs.

    A régi csomagokban még nincs ilyen mező: ott `not_collected`, hogy a
    felület ne állítsa, hogy „a lánc nem volt likvid”, amikor meg sem néztük.
    """
    status = row.get("implied_status")
    if not isinstance(status, str):
        return {"status": "not_collected"}
    if status != "ok":
        return {"status": status}
    expiry = row.get("implied_expiry")
    return {
        "status": "ok",
        "prob_up": _num(row.get("implied_prob")),
        "iv": _num(row.get("implied_iv")),
        "band_low": _num(row.get("implied_band_low")),
        "band_high": _num(row.get("implied_band_high")),
        "expiry": str(expiry) if expiry is not None and pd.notna(expiry) else None,
    }


#: Legalább ennyi lezárt becslés kell egy papír kalibráció-minőségéhez.
SCREENER_MIN_RESOLVED = 30
#: Ennyi napos mini árfolyamgörbe kerül a screener soraiba.
SPARK_SESSIONS = 60


def screener_row(
    meta: dict[str, object],
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict[str, object]:
    """Egy papír sora a screenerben (`docs/postmortem-es-screener.md`, 3.).

    A kalibráció-minőség a Brier-készség a naiv baseline-hoz, az összes lezárt
    becslésen: `1 − Brier(modell) / Brier(baseline)`. 30 lezárt becslés alatt
    nincs értéke — a sorban csak a mintaszám áll.
    """
    bars = prices.sort_values("date").tail(SPARK_SESSIONS)
    f20 = forecasts[forecasts["horizon"] == 20] if "horizon" in forecasts else pd.DataFrame()
    first = f20.iloc[0] if not f20.empty else None
    n = len(outcomes)
    skill = None
    if n >= SCREENER_MIN_RESOLVED:
        base = float(outcomes["baseline_brier"].mean())
        if base > 0:
            skill = round(1 - float(outcomes["brier"].mean()) / base, 4)
    return {
        **meta,
        "spark": [_num(v, 4) for v in bars["close"]],
        "prob_up": _num(first["prob_up"]) if first is not None else None,
        "baseline_prob": _num(first["baseline_prob"]) if first is not None else None,
        "band_low": _num(first["band_low"]) if first is not None else None,
        "band_high": _num(first["band_high"]) if first is not None else None,
        "regime": str(first["regime"]) if first is not None and pd.notna(first.get("regime")) else None,
        "resolved": n,
        "calibration_skill": skill,
    }


def build_history(prices: pd.DataFrame) -> dict[str, object]:
    """A munkaasztal idősora egy papírra, oszlopos formában.

    Soronkénti objektumok helyett oszloponként egy-egy tömb: 1460 napnál ez a
    fájlméret nagyjából harmada, és a felület úgyis oszloponként számol
    belőle indikátort.
    """
    bars = prices.sort_values("date").tail(WORKBENCH_SESSIONS)
    return {
        "d": [str(v) for v in bars["date"]],
        "o": [_num(v, 4) for v in bars["open"]],
        "h": [_num(v, 4) for v in bars["high"]],
        "l": [_num(v, 4) for v in bars["low"]],
        "c": [_num(v, 4) for v in bars["close"]],
        "v": [_num(v, 0) for v in bars["volume"]],
    }


def build_instrument(
    meta: dict[str, object],
    prices: pd.DataFrame,
    forecasts: pd.DataFrame,
    history: pd.DataFrame,
    outcomes: pd.DataFrame,
    session: date,
) -> dict[str, object]:
    """Egy papír teljes megjelenítési csomagja."""
    bars = prices.sort_values("date").tail(HISTORY_SESSIONS)
    candles = [
        {
            "d": str(row["date"]),
            "o": _num(row.get("open"), 4),
            "h": _num(row.get("high"), 4),
            "l": _num(row.get("low"), 4),
            "c": _num(row.get("close"), 4),
            "v": _num(row.get("volume"), 0),
        }
        for row in bars.to_dict("records")
    ]

    close = candles[-1]["c"] if candles else None
    previous = candles[-2]["c"] if len(candles) > 1 else None
    change = None
    if close is not None and previous not in (None, 0):
        change = _num(close / float(previous) - 1.0)

    # Van papír, amire nem készül becslés (rövid múlt, friss leválasztás). A
    # csomagja akkor is elkészül, csak becslés nélkül — a felület ezt kiírja.
    horizons = []
    ordered = forecasts.sort_values("horizon") if "horizon" in forecasts else forecasts
    for row in ordered.to_dict("records"):
        horizons.append(format_forecast(row))

    timeline_rows = (
        history.sort_values(["session", "horizon"])
        if {"session", "horizon"}.issubset(history.columns)
        else history
    )
    timeline = [
        {
            "session": str(row["session"]),
            "horizon": int(row["horizon"]),
            "prob_up": _num(row.get("prob_up")),
            "band_low": _num(row.get("band_low")),
            "band_high": _num(row.get("band_high")),
        }
        for row in timeline_rows.to_dict("records")
    ]

    # A papírra szűrt élő teljesítmény: amíg nincs 30 lezárt megfigyelés,
    # csak a mintaszám megy ki — százalék nem (2. sarokkő).
    resolved = len(outcomes)
    performance = {
        "n": int(resolved),
        "min_observations": MIN_OBSERVATIONS,
        "hits": int(outcomes["hit"].sum()) if resolved else 0,
        "baseline_hits": int(outcomes["baseline_hit"].sum()) if resolved else 0,
    }
    # A piaci árazás ellen csak ott mérünk, ahol elérhető volt: külön mintaszám.
    if resolved and "implied_hit" in outcomes:
        paired = outcomes[outcomes["implied_hit"].notna()]
        performance["implied_n"] = len(paired)
        performance["implied_hits"] = int(paired["implied_hit"].sum())
        performance["implied_model_hits"] = int(paired["hit"].sum())
    else:
        performance["implied_n"] = 0

    return {
        "instrument": meta,
        "session": str(session),
        "close": close,
        "change": change,
        "regime": str(forecasts["regime"].iloc[0]) if not forecasts.empty and "regime" in forecasts else None,
        "candles": candles,
        "forecasts": horizons,
        "timeline": timeline,
        "performance": performance,
    }


#: A felület fő modellje; a többi kihívó (docs/modell-arena.md).
MAIN_FAMILY = "lgbm-core"


def _arena_rows(table: pd.DataFrame | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if table is None or table.empty:
        return rows
    for r in table.to_dict("records"):
        needed = r.get("observations_needed")
        rows.append(
            {
                "family": r["family"],
                "version": r["version"],
                "horizon": int(r["horizon"]),
                "metric": r["metric"],
                "against": r["against"],
                "value": _num(r["value"]),
                "baseline_value": _num(r["baseline_value"]),
                "delta": _num(r["delta"]),
                "n": int(r["n"]),
                "n_eff": _num(r["n_eff"], 1),
                "p_value_fdr": _num(r["p_value_fdr"]),
                # soronként, mert az 1 és a 120 nap külön FDR-családban van
                "n_tests": int(r.get("n_tests") or 0),
                "verdict": r["verdict"],
                "observations_needed": None if needed is None or pd.isna(needed) else int(needed),
                "first_observed": r["first_observed"],
                "last_observed": r["last_observed"],
            }
        )
    return rows


def build_model_arena(
    table: pd.DataFrame | None, now: datetime, backtest: pd.DataFrame | None = None
) -> dict[str, object]:
    """A modell-aréna megjelenítése (`docs/modell-arena.md`).

    Amíg nincs lezárt kihívó-becslés, a csomag üres sorokkal megy ki: a felület
    ebből mondja ki, hogy az élő rekord még nem indult — nem hallgat el róla.
    """
    rows = _arena_rows(table)
    return {
        "generated_at": now.astimezone(UTC).replace(microsecond=0).isoformat(),
        "main": MAIN_FAMILY,
        "families": ["lgbm-core", "ar-linear", "mlp-core", "ensemble"],
        "n_tests": int(table["n_tests"].iloc[0]) if table is not None and not table.empty else 0,
        "rows": rows,
        # A backtest külön kulcs alatt: a felület soha nem keverheti az élő számmal.
        "backtest": {
            "n_tests": int(backtest["n_tests"].iloc[0]) if backtest is not None and not backtest.empty else 0,
            "rows": _arena_rows(backtest),
        },
    }


def build_news_arena(table: pd.DataFrame | None, now: datetime) -> dict[str, object]:
    """A hír-aréna megjelenítése (`docs/hir-arena.md`): élő és backtest külön kulcs alatt."""

    def rows(period: str) -> list[dict[str, object]]:
        if table is None or table.empty:
            return []
        out = []
        for r in table[table["period"] == period].to_dict("records"):
            needed = r.get("observations_needed")
            out.append(
                {
                    "signal": r["signal"],
                    "value": _num(r["value"]),
                    "baseline_value": _num(r["baseline_value"]),
                    "delta": _num(r["delta"]),
                    "n": int(r["n"]),
                    "n_eff": _num(r["n_eff"], 1),
                    "p_value_fdr": _num(r["p_value_fdr"]),
                    "verdict": r["verdict"],
                    "observations_needed": None if needed is None or pd.isna(needed) else int(needed),
                    "first": r["first"],
                    "last": r["last"],
                }
            )
        return out

    return {
        "generated_at": now.astimezone(UTC).replace(microsecond=0).isoformat(),
        "live_from": "2026-09-26",
        "live": rows("live"),
        "backtest": rows("backtest"),
    }


def build_lab(table: pd.DataFrame | None, now: datetime) -> dict[str, object]:
    """A labor rácsa (`docs/labor.md`). Minden pont kimegy, a gyengék is."""
    rows: list[dict[str, object]] = []
    if table is not None and not table.empty:
        for r in table.to_dict("records"):
            needed = r.get("observations_needed")
            rows.append(
                {
                    "universe": r["universe"],
                    "horizon": int(r["horizon"]),
                    "threshold": float(r["threshold"]),
                    "cost": float(r["cost"]),
                    "periods": int(r["periods"]),
                    "first": r["first"],
                    "last": r["last"],
                    **{
                        k: _num(r[k])
                        for k in (
                            "ann_return",
                            "ann_vol",
                            "max_drawdown",
                            "bh_ann_return",
                            "bh_ann_vol",
                            "bh_max_drawdown",
                            "share_better",
                            "avg_positions",
                            "excess_per_period",
                            "n_eff",
                            "p_value_fdr",
                        )
                    },
                    "cash_periods": int(r["cash_periods"]),
                    "n_tests": int(r.get("n_tests") or 0),
                    "verdict": r["verdict"],
                    "observations_needed": None if needed is None or pd.isna(needed) else int(needed),
                    "curve": json.loads(r["curve"]) if isinstance(r["curve"], str) else [],
                }
            )
    return {
        "generated_at": now.astimezone(UTC).replace(microsecond=0).isoformat(),
        "n_tests": int(table["n_tests"].iloc[0]) if table is not None and not table.empty else 0,
        "rows": rows,
    }


def run(storage: Storage, now: datetime, dry_run: bool = False) -> dict[str, object]:
    session = last_closed_session(now)
    years = sorted({2026, session.year})

    all_forecasts = load_forecast_history(storage, years)
    if all_forecasts.empty:
        log.info("publish_nothing", reason="még nincs lementett becslés")
        return {"session": str(session), "instruments": 0, "status": "no_forecasts"}

    latest_session = max(all_forecasts["session"])
    today_forecasts = all_forecasts[all_forecasts["session"] == latest_session]
    outcomes = load_outcomes(storage, years)
    arena = _read_table(storage, LIVE_ARENA_PATH)
    arena_records = build_arena(arena if arena is not None else pd.DataFrame())

    # Az univerzum tagsága a MAI napra értendő, nem a becslés sessionjére: a
    # lista 2026-09-19-én készült, az első becslés viszont az előző napra szól,
    # és akkor minden papír kiesne a szűrőből.
    universe = active_on(load_universe(), now.astimezone(UTC).date())
    # A munkaasztalnak hosszabb múlt kell, mint a többi képernyőnek. Egyszer
    # töltjük be, és a meglévő építők ugyanazt a hároméves szeletet kapják,
    # mint eddig — a viselkedésük nem változhat attól, hogy a munkaasztal
    # bekerült.
    long_prices = _load_prices(
        storage, list(range(max(2005, latest_session.year - WORKBENCH_YEARS + 1), latest_session.year + 1))
    )
    prices = long_prices[pd.to_datetime(long_prices["date"]).dt.year >= latest_session.year - 2]
    regime = _read_table(storage, REGIME_PATH)

    cutoff = pd.Timestamp(latest_session) - pd.Timedelta(days=TIMELINE_SESSIONS * 2)
    timeline_source = all_forecasts[pd.to_datetime(all_forecasts["session"]) >= cutoff]

    latest_doc = build_latest(
        latest_session,
        universe,
        today_forecasts,
        all_forecasts,
        outcomes,
        arena_records,
        regime,
        now,
    )
    # Hivatalos közlemények címkeként, csak sokk vagy esemény napján (docs/hirek.md).
    if not dry_run:
        latest_doc["news"] = _news(latest_session, latest_doc)
    files: list[tuple[str, bytes]] = [
        ("latest.json", _dumps(latest_doc)),
        ("arena.json", _dumps(arena_records)),
        ("honesty.json", _dumps(build_honesty(prices, latest_session))),
        ("instruments.json", _dumps(build_index(universe, today_forecasts, prices))),
    ]

    # A konfluencia élő panelje (docs/konfluencia.md, 5.): a mai kiváltók és
    # kontextusok, a heti mérés számaival. Ha bármi elbukik, a napi csomag
    # ettől nem áll meg; a panel kimondja, hogy ma nem érhető el.
    # Itt töltjük be, nem a modul tetején: a konfluencia a publikálás állandóit
    # használja, a felső import körbeérne.
    from pipeline.confluence.live import panel, standing_all
    from pipeline.confluence.run import RESULTS_PATH as CONFLUENCE_RESULTS_PATH

    confluence_results = _read_table(storage, CONFLUENCE_RESULTS_PATH)
    try:
        regime_labels: dict[date, str] = {}
        if regime is not None and not regime.empty:
            labelled = regime.dropna(subset=["regime"])
            regime_labels = dict(
                zip(pd.to_datetime(labelled["date"]).dt.date, labelled["regime"].astype(str), strict=True)
            )
        standing_today = standing_all(long_prices, regime_labels, latest_session)
        confluence_ok = True
    except Exception as error:  # noqa: BLE001 — a napi csomag fontosabb a panelnél
        log.warning("confluence_panel_failed", error=type(error).__name__)
        standing_today, confluence_ok = {}, False

    by_instrument = {i: g for i, g in today_forecasts.groupby("instrument_id")}
    price_groups = {i: g for i, g in prices.groupby("instrument_id")}
    long_groups = {i: g for i, g in long_prices.groupby("instrument_id")}
    timeline_groups = {i: g for i, g in timeline_source.groupby("instrument_id")}
    outcome_groups = {i: g for i, g in outcomes.groupby("instrument_id")} if not outcomes.empty else {}

    written = 0
    screener: list[dict[str, object]] = []
    for row in universe.to_dict("records"):
        instrument_id = str(row["instrument_id"])
        payload = build_instrument(
            meta={
                "id": instrument_id,
                "ticker": row["ticker"],
                "name": row["name"],
                "asset_class": row.get("asset_class"),
                "sector": row.get("sector"),
                "exchange_calendar": row.get("exchange_calendar"),
            },
            prices=price_groups.get(instrument_id, pd.DataFrame(columns=prices.columns)),
            forecasts=by_instrument.get(instrument_id, pd.DataFrame()),
            history=timeline_groups.get(instrument_id, pd.DataFrame(columns=["session", "horizon"])),
            outcomes=outcome_groups.get(instrument_id, pd.DataFrame()),
            session=latest_session,
        )
        payload["confluence"] = (
            panel(standing_today.get(instrument_id, []), confluence_results, latest_session)
            if confluence_ok
            else {
                "session": str(latest_session),
                "measured": False,
                "available": False,
                "tested": 0,
                "active": [],
            }
        )
        files.append((f"instruments/{instrument_id}.json", _dumps(payload)))
        screener.append(
            screener_row(
                {
                    "id": instrument_id,
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "sector": row.get("sector"),
                    "asset_class": row.get("asset_class"),
                },
                price_groups.get(instrument_id, pd.DataFrame(columns=prices.columns)),
                by_instrument.get(instrument_id, pd.DataFrame()),
                outcome_groups.get(instrument_id, pd.DataFrame(columns=["brier", "baseline_brier"])),
            )
        )
        long_bars = long_groups.get(instrument_id)
        if long_bars is not None and not long_bars.empty:
            files.append((f"history/{instrument_id}.json", _dumps(build_history(long_bars))))
        written += 1

    files.append(
        (
            "screener.json",
            _dumps({"min_resolved": SCREENER_MIN_RESOLVED, "session": str(latest_session), "rows": screener}),
        )
    )
    files.append(("postmortems.json", _dumps(build_postmortems(outcomes, universe))))
    # A teljes eredménynapló (docs/eredmenynaplo.md). A havi letöltések csak
    # akkor mennek fel újra, ha változtak; a jegyzék a record.json-ban mindig teljes.
    # A tickerhez a teljes lista kell: a kivezetett papír sora sem maradhat név nélkül.
    from pipeline.publish.record import EXPORT_PREFIX, build_record

    existing_exports = set() if dry_run else set(storage.list(DISPLAY_BUCKET, EXPORT_PREFIX))
    record_doc, record_files = build_record(outcomes, all_forecasts, load_universe(), existing_exports, now)
    files.append(("record.json", _dumps(record_doc)))
    from pipeline.model.run import ARENA_BACKTEST_PATH, LAB_PATH

    files.append(("lab.json", _dumps(build_lab(_read_table(storage, LAB_PATH), now))))
    from pipeline.newsarena.run import RESULTS_PATH as NEWS_ARENA_PATH

    files.append(("news-arena.json", _dumps(build_news_arena(_read_table(storage, NEWS_ARENA_PATH), now))))
    from pipeline.resolve.arena import MODELS_LIVE_PATH

    files.append(
        (
            "model-arena.json",
            _dumps(
                build_model_arena(
                    _read_table(storage, MODELS_LIVE_PATH), now, _read_table(storage, ARENA_BACKTEST_PATH)
                )
            ),
        )
    )

    if dry_run:
        log.info("publish_dry_run", files=len(files), exports=len(record_files), instruments=written)
        return {"session": str(latest_session), "instruments": written, "status": "dry_run"}

    storage.ensure_private_bucket(DISPLAY_BUCKET)
    from pipeline.publish.manifest import ChangedOnly

    # Csak az új vagy eltérő fájl megy fel (docs: privát repó, terv-egress.md, J3).
    uploader = ChangedOnly(storage, DISPLAY_BUCKET, MANIFEST_PATH)
    try:
        for path, blob in files:
            uploader.put(path, blob, "application/json")
        for path, blob, content_type in record_files:
            uploader.put(path, blob, content_type)
    finally:
        uploader.finish()

    log.info("publish_done", session=str(latest_session), files=len(files), exports=len(record_files))
    return {
        "session": str(latest_session),
        "instruments": written,
        "files": len(files),
        "status": "published",
        "latest": latest_doc,
    }


def main(argv: list[str] | None = None) -> int:
    logging_setup.configure()
    parser = argparse.ArgumentParser(description="Megjelenítési JSON-ok a webappnak")
    parser.add_argument("--local", type=Path, help="helyi tár (próbafuttatáshoz)")
    parser.add_argument("--dry-run", action="store_true", help="ne töltsön fel semmit")
    parser.add_argument("--out", type=Path, help="a JSON-ok mentése ide (próbafuttatáshoz)")
    args = parser.parse_args(argv)

    if args.local is not None:
        storage: Storage = LocalStorage(args.local)
    else:
        settings = load_settings()
        if not settings.storage_configured:
            print("Hiányzik a SUPABASE_URL vagy a SUPABASE_SECRET_KEY.", file=sys.stderr)
            return 2
        storage = SupabaseStorage(settings.supabase_url or "", settings.supabase_secret_key or "")

    from pipeline.runstate import gated

    if args.dry_run:
        result = run(storage, datetime.now(UTC), dry_run=True)
    else:
        gated_result = gated(
            storage, "publish-equity", lambda: run(storage, datetime.now(UTC), dry_run=False)
        )
        if gated_result is None:
            print(json.dumps({"status": "skipped"}))
            return 0
        result = gated_result
    latest = result.pop("latest", None)
    if args.local is None and not args.dry_run and isinstance(latest, dict):
        # Az értesítések piaci összefoglalója (spec/07, 6.). A felhasználókra
        # bontás az adatbázisban történik; ide felhasználói adat nem jön vissza.
        from pipeline.notify import enqueue, push_calendars, push_sessions, push_target_days

        settings = load_settings()
        enqueue(settings.supabase_url or "", settings.supabase_secret_key or "", latest)
        # A kereskedési naptár a „ma zárul a tézised” értesítéshez (0021).
        push_sessions(
            settings.supabase_url or "", settings.supabase_secret_key or "", datetime.now(UTC).date()
        )
        # Melyik papír melyik naptárat követi (0022): a kripto-tézis zárónapjához.
        push_calendars(settings.supabase_url or "", settings.supabase_secret_key or "")
        # A TARGET-napok a deviza-tézis zárónapjához (0024).
        push_target_days(
            settings.supabase_url or "", settings.supabase_secret_key or "", datetime.now(UTC).date()
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
