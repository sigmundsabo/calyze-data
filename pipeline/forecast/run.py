"""Napi becslés: a megjelenítés ELŐTT lementve (spec/06, 1. lépés).

A termék egyetlen visszamenőleg nem hamisítható állítása az idő. Ezért:

- a napi becslés-csomag a privát tárba kerül, és **soha nem íródik felül**: ha
  az adott sessionre már van csomag, a futás nem ír, hanem kilép;
- a csomag SHA-256 lenyomata egy kis manifestben a **publikus repóba**
  commitolódik. A commit időbélyege bizonyítja, hogy a becslés akkor már
  megvolt, és bárki újraszámolhatja a hash-t, amikor a csomag nyilvánossá
  válik (az adatlicenc rendezése után; lásd `docs/adatlicencek.md` a másik
  repóban);
- a becslés azonosítója determinisztikus (`uuid5`), így ugyanarra a napra és
  papírra kétszer nem keletkezhet két különböző azonosítójú sor.

Futtatás:
    uv run python -m pipeline.forecast.run
    uv run python -m pipeline.forecast.run --local data --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import log as logging_setup
from pipeline.calendar import last_closed_session, session_lag, sessions_back
from pipeline.config import RAW_BUCKET, load_settings
from pipeline.features.run import (
    MACRO_PATH,
    REGIME_PATH,
    _load_prices,
    _read_table,
    build_features,
)
from pipeline.ingest.partitions import ACTIONS_PATH
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.model.baselines import add_sector_return
from pipeline.model.config import CORE_HORIZONS, HORIZONS, MIN_HISTORY_SESSIONS, MODEL_FAMILY, MODEL_VERSION
from pipeline.model.loading import load_horizon_models
from pipeline.model.run import MODEL_PREFIX
from pipeline.universe import active_on, load_universe

log = logging_setup.get_logger(__name__)

FORECAST_PREFIX = "forecasts"
#: A becslés-azonosítók névtere (rögzített, hogy az azonosító reprodukálható legyen).
NAMESPACE = uuid.UUID("4d1f0d9e-2a3b-5c6d-8e7f-000000000001")
#: Ennyi év árfolyamát tölti le a napi futás (az EMA-200 és a 252 napos momentum miatt).
LOOKBACK_YEARS = 3

# Ennyi kihagyott kereskedési napig még becslünk a legfrissebb meglévő adatból.
# Fölötte a forrás nem „késik", hanem kiesett: olyankor a hallgatás az őszinte
# válasz, nem egy elavult adatra épített szám.
MAX_SOURCE_LAG_SESSIONS = 3

# Egy nap akkor számít mérhetőnek, ha az univerzum ekkora hányadára van adat.
# A forrás néha egyetlen papírra már közzéteszi az aznapi sort, a többire még
# nem; egy ilyen napra „napi mérést" mondani félrevezető lenne.
MIN_SESSION_COVERAGE = 0.8


def forecast_id(instrument_id: str, session: date, horizon: int, version: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{instrument_id}|{session.isoformat()}|{horizon}|{version}"))


def target_session(session: date, horizon: int, sessions: list[date]) -> date | None:
    """A horizont vége: az N-edik session zárása a becslés napja után."""
    try:
        i = sessions.index(session)
    except ValueError:
        return None
    return sessions[i + horizon] if i + horizon < len(sessions) else None


#: A modell-aréna kihívóinak csomagja (`docs/modell-arena.md`, 2.): külön a fő csomagtól.
ARENA_PREFIX = "forecasts-arena"


def arena_package_path(session: date) -> str:
    return f"{ARENA_PREFIX}/{session.year}/{session.isoformat()}.parquet"


def package_path(session: date) -> str:
    return f"{FORECAST_PREFIX}/{session.year}/{session.isoformat()}.parquet"


def package_exists(storage: Storage, session: date) -> bool:
    """Van-e már csomag a napra, letöltés nélkül (a mappa listájából; J2)."""
    path = package_path(session)
    return path in set(storage.list(RAW_BUCKET, path.rsplit("/", 1)[0]))


def build_forecasts(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    session: date,
    models: dict[int, object],
    future_sessions: list[date],
    made_at: datetime,
) -> pd.DataFrame:
    """A session napi becslések minden lefedett papírra, horizontonként."""
    today = features[features["date"] == session]
    today = today[today["history_sessions"] >= MIN_HISTORY_SESSIONS]
    if today.empty:
        raise RuntimeError(f"Nincs feature-sor a(z) {session} sessionre.")

    last_close = (
        prices[prices["date"] == session].set_index("instrument_id")["close"].astype("float64").to_dict()
    )
    rows: list[pd.DataFrame] = []
    for horizon in sorted(models):
        bundle = models[horizon]
        model = bundle["model"]  # type: ignore[index]
        naive = bundle["baselines"]["naive"]  # type: ignore[index]
        out = model.predict(today)
        # A magyarázat a csomag része, tehát a napi lenyomat fedi: utólag nem
        # gyártható le másként, mint ahogy a becslés készült.
        out["contributions"] = [json.dumps(c, separators=(",", ":")) for c in model.contributions(today)]
        # A horizontot itt állítjuk be, nem a modelltől vesszük: a csomag
        # szerkezete akkor is helyes marad, ha egy modell rosszul tölti ki.
        out["horizon"] = horizon
        base = naive.predict(today)
        out["baseline_prob"] = base["prob_up"].to_numpy()
        out["baseline_id"] = "naive"
        out["session"] = session
        out["target_session"] = target_session(session, horizon, future_sessions)
        out["made_at"] = pd.Timestamp(made_at)
        out["model_family"] = MODEL_FAMILY
        out["model_version"] = MODEL_VERSION
        out["regime"] = today["regime"].to_numpy()
        # A sáv árban is: a záróárra vetítve (a log teljes hozam sávjából).
        close = np.array([last_close.get(i, np.nan) for i in out["instrument_id"]])
        out["close"] = close
        out["price_low"] = close * np.exp(out["band_low"].to_numpy())
        out["price_high"] = close * np.exp(out["band_high"].to_numpy())
        out["expected_price"] = close * np.exp(out["expected_return"].to_numpy())
        out["forecast_id"] = [forecast_id(i, session, horizon, MODEL_VERSION) for i in out["instrument_id"]]
        rows.append(out)

    frame = pd.concat(rows, ignore_index=True)
    return frame.drop(columns=["prob_up_raw"], errors="ignore")


def build_challenger_forecasts(
    features: pd.DataFrame,
    session: date,
    challengers: dict[str, dict[int, object]],
    main: pd.DataFrame,
    made_at: datetime,
) -> pd.DataFrame:
    """A kihívó családok becslései ugyanarra a napra és papírokra, mint a fő csomag.

    A baseline, a célnap és a rezsim a fő csomagból jön, soronként párosítva:
    így a kihívó és az `lgbm-core` ugyanazon a soron, ugyanahhoz a baseline-hoz
    mérhető (`docs/modell-arena.md`, 2.).
    """
    today = features[features["date"] == session]
    today = today[today["history_sessions"] >= MIN_HISTORY_SESSIONS]
    keys = main[["instrument_id", "horizon", "baseline_prob", "target_session", "regime"]]
    rows = []
    for family, by_horizon in sorted(challengers.items()):
        for horizon, model in sorted(by_horizon.items()):
            out = model.predict(today)  # type: ignore[attr-defined]
            out["horizon"] = horizon
            out = out.merge(keys, on=["instrument_id", "horizon"], how="inner")
            version = str(getattr(model, "version", "v1"))
            out["session"] = session
            out["made_at"] = pd.Timestamp(made_at)
            out["model_family"] = family
            out["model_version"] = version
            out["forecast_id"] = [
                forecast_id(i, session, horizon, f"{family}-{version}") for i in out["instrument_id"]
            ]
            rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop(columns=["date"], errors="ignore")


def load_challengers(storage: Storage) -> dict[str, dict[int, object]]:
    """A kihívók betanított modelljei. Ami hiányzik, az kimarad (a napon „not available”).

    Egy család akkor él, ha a régi három időtáv megvan; a bővített időtávok az
    első tanításig hiányozhatnak, ahogy a fő modellnél is.
    """
    from pipeline.model.families import FAMILY_VERSION

    out: dict[str, dict[int, object]] = {}
    for family in ("ar-linear", "mlp-core", "ensemble"):
        models = {}
        for horizon in HORIZONS:
            blob = storage.download(RAW_BUCKET, f"models/{family}/{FAMILY_VERSION}/h{horizon}.pkl")
            if blob is not None:
                models[horizon] = pickle.loads(blob)  # noqa: S301 — saját, privát tárból származó fájl
        if all(h in models for h in CORE_HORIZONS):
            out[family] = models
    return out


def load_models(storage: Storage) -> dict[int, object]:
    return load_horizon_models(storage, MODEL_PREFIX, "python -m pipeline.model.run --task train")  # type: ignore[return-value]


def manifest_entry(
    session: date, package: bytes, frame: pd.DataFrame, made_at: datetime, universe: int
) -> dict[str, object]:
    """A publikus repóba kerülő kis manifest: lenyomat, méret, darabszámok.

    Az univerzum mérete is bekerül, hogy a lefedettség utólag is értelmezhető
    legyen: egy 1 papíros nap önmagában nem árulja el, hogy 620-ból egy volt.
    """
    return {
        "session": session.isoformat(),
        "made_at": made_at.isoformat(timespec="seconds"),
        "model_family": MODEL_FAMILY,
        "model_version": MODEL_VERSION,
        "horizons": sorted(int(h) for h in frame["horizon"].unique()),
        "instruments": int(frame["instrument_id"].nunique()),
        "universe": universe,
        "forecasts": len(frame),
        "sha256": hashlib.sha256(package).hexdigest(),
        "bytes": len(package),
        "package": "private storage until the data licence allows publication",
    }


def risk_free_rate(macro: pd.DataFrame | None, session: date) -> float | None:
    """A 3 hónapos állampapír-hozam a becslés napján vagy előtte (`DGS3MO`).

    A FRED százalékban adja; itt évesített tizedes tört. `None`, ha nincs.
    """
    if macro is None or "yield_3m" not in macro:
        return None
    series = macro["yield_3m"].dropna()
    series = series[pd.to_datetime(series.index).date <= session]
    return None if series.empty else float(series.iloc[-1]) / 100


def implied_baseline(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    macro: pd.DataFrame | None,
    session: date,
    future: list[date],
) -> pd.DataFrame:
    """A piac-implikált baseline minden papírra (`docs/piac-implikalt-baseline.md`).

    A becslés ELŐTT kérjük le, és a becslés-csomagba kerül: a napi lenyomat
    ezt is fedi. Ha a nem hivatalos forrás elhal, üres táblát adunk — a
    becslés ettől még elkészül.
    """
    from pipeline.options.fetch import COLUMNS, fetch_implied

    try:
        spots = (
            prices[prices["date"] == session].set_index("instrument_id")["close"].astype("float64").to_dict()
        )
        targets = {h: target_session(session, h, future) for h in HORIZONS}
        instruments = list(zip(universe["instrument_id"], universe["ticker"], strict=True))
        return fetch_implied(instruments, spots, session, targets, risk_free_rate(macro, session))
    except Exception as error:  # noqa: BLE001 — a forrás kiesése nem állíthatja meg a mérést
        log.warning("implied_baseline_failed", error=type(error).__name__)
        return pd.DataFrame(columns=COLUMNS)


def attach_shock_flags(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    actions: pd.DataFrame,
    universe: pd.DataFrame,
    macro: pd.DataFrame | None,
    session: date,
) -> pd.DataFrame:
    """A sokk-detektor jelei és a visszatartás (docs/sokk-detektor.md).

    A detektor hibája nem állíthatja meg a becslést: ilyenkor a státusz
    `failed`, és a felület nem állítja, hogy nem volt jelzés.
    """
    from pipeline.shocks.detect import attach_shocks, detect

    try:
        classified, shock = detect(prices, actions, universe, macro, session)
        out = attach_shocks(frame, classified, shock)
        out["shock_status"] = "ok"
        log.info(
            "shock_checked",
            instruments_flagged=int((classified["signals"] != "").sum()) if not classified.empty else 0,
            market=",".join(shock.signals),
            withheld=shock.withheld,
        )
        return out
    except Exception as error:  # noqa: BLE001 — a detektor kiesése nem állíthatja meg a mérést
        log.warning("shock_detect_failed", error=type(error).__name__)
        out = frame.copy()
        out["shock_status"] = "failed"
        out["withheld"] = False
        return out


def too_late(now: datetime, next_open: datetime | None) -> bool:
    """Elkésett-e a becslés: a napja utáni első tőzsdenap már kinyitott.

    Ugyanaz az elv, mint kriptón és devizán (6 órás szabály): a lenyomatnak
    azelőtt kell rögzülnie, hogy az ablakából bármi ismert lenne.
    """
    return next_open is None or now.astimezone(UTC) >= next_open.astimezone(UTC)


def partial_package(frame: pd.DataFrame, universe_size: int) -> bool:
    """Részleges-e a kész csomag (a lefedettségi küszöb a TÉNYLEGES becsléseken)."""
    covered = int(frame["instrument_id"].nunique()) if not frame.empty else 0
    return covered < max(int(universe_size * MIN_SESSION_COVERAGE), 1)


def implied_allowed(now: datetime, next_open: datetime | None) -> bool:
    """Szabad-e még opciós árat lekérni a becslés napjához.

    Csak addig, amíg a becslés napja utáni első tőzsdenap ki nem nyitott.
    Ha a becslés késve készül (a forrás egy napot kihagyott), vagy valaki
    kereskedési időben indítja kézzel, a lánc már a becslés napja UTÁNI
    tudást hordozná — ez szivárgás, tehát inkább „nem elérhető".
    """
    return next_open is not None and now.astimezone(UTC) < next_open.astimezone(UTC)


def attach_implied(frame: pd.DataFrame, implied: pd.DataFrame, missing: str = "fetch_failed") -> pd.DataFrame:
    """Az implikált oszlopok a becslés-sorokhoz. Ahol nincs sor: `missing`."""
    from pipeline.options.fetch import COLUMNS

    if implied.empty:
        implied = pd.DataFrame(columns=COLUMNS)
    merged = frame.merge(implied, on=["instrument_id", "horizon"], how="left")
    merged["implied_status"] = merged["implied_status"].fillna(missing)
    return merged


def choose_session(coverage: Mapping[date, int], expected: date, universe_size: int) -> date:
    """Melyik napra szóljon a becslés: az ADAT napjára, nem a naptáréra.

    A naptár tudja, mikor volt tőzsdenap; az ingyenes forrás viszont néha
    egy-egy napot kihagy, vagy késve teszi közzé. Ha a naptárhoz
    ragaszkodnánk, a futás elszállna, és aznap nem mérnénk semmit — pedig a
    meglévő adatra lehetne becslést adni.

    Nem elég azonban, hogy a nap LÉTEZZEN az adatban: elég papírra kell
    léteznie. A forrás néha egyetlen papírra már közzéteszi az aznapi sort, a
    többire még nem; ilyenkor a „legfrissebb nap" egy papírt jelentene 620
    helyett, és a napi mérés hazugság lenne.

    Amit nem csinálunk: nem találunk ki árat a hiányzó napra, és nem
    tüntetjük fel a becslést frissebbnek, mint az adat, amiből készült.

    `MAX_SOURCE_LAG_SESSIONS` fölött megállunk: annyi kihagyás után a forrás
    nem késik, hanem kiesett, és a hallgatás az őszinte válasz.
    """
    need = max(int(universe_size * MIN_SESSION_COVERAGE), 1)
    usable = sorted(day for day, count in coverage.items() if count >= need and day <= expected)
    if not usable:
        best = max(coverage.values(), default=0)
        raise RuntimeError(
            f"Egyetlen napra sincs elég adat: a legjobb nap {best} papírt fed le, "
            f"a küszöb {need} ({universe_size} papír {MIN_SESSION_COVERAGE:.0%}-a)."
        )
    available = usable[-1]
    lag = session_lag(available, expected)
    if lag > MAX_SOURCE_LAG_SESSIONS:
        raise RuntimeError(
            f"A forrás {lag} kereskedési nappal marad el: a legfrissebb használható nap "
            f"{available}, az utolsó zárt nap {expected}. Ennyi kihagyás után nem becslünk."
        )
    if lag > 0:
        log.info(
            "forecast_stale_source",
            session=str(available),
            expected=str(expected),
            lag=lag,
            instruments=coverage[available],
        )
    return available


def run(storage: Storage, now: datetime, dry_run: bool = False) -> dict[str, object]:
    expected = last_closed_session(now)
    # A „már lementettük?" kérdést csak a session KIVÁLASZTÁSA után lehet
    # feltenni: a naptár szerinti napra lehet, hogy soha nem is becslünk,
    # mert nincs rá elég adat. A korai kilépés egy percnyi számítást spórolt
    # volna, cserébe egy rossz napra nézett volna rá.

    # Ha viszont a naptár szerinti napra már van csomag, nincs mit választani:
    # a napi újrapróbák így az árfolyam letöltése nélkül lépnek ki (J2).
    if package_exists(storage, expected):
        log.info("forecast_already_saved", session=str(expected))
        return {"session": str(expected), "status": "already_saved"}

    universe = active_on(load_universe(), now.astimezone(UTC).date())
    years = list(range(expected.year - LOOKBACK_YEARS + 1, expected.year + 1))
    prices = _load_prices(storage, years)
    actions = _read_table(storage, ACTIONS_PATH)
    if actions is None:
        actions = pd.DataFrame(columns=["instrument_id", "date", "dividend", "split_ratio"])
    regime = _read_table(storage, REGIME_PATH)
    macro = _read_table(storage, MACRO_PATH)
    if regime is None:
        raise RuntimeError("Nincs rezsim-tábla — előbb a feature-futás kell.")
    if macro is not None:
        macro = macro.set_index("date")

    features = add_sector_return(build_features(prices, actions, universe, regime, macro))

    coverage = features.groupby("date")["instrument_id"].nunique().to_dict()
    session = choose_session(coverage, expected, len(universe))

    if package_exists(storage, session):
        log.info("forecast_already_saved", session=str(session))
        return {"session": str(session), "status": "already_saved"}

    # A horizont végéhez a jövőbeli kereskedési napok is kellenek: a naptárból,
    # nem az árfolyamból (az még nem létezik).
    from pipeline.calendar import calendar

    cal = calendar()
    future = [
        ts.date()
        for ts in cal.sessions_in_range(pd.Timestamp(sessions_back(session, 2)[0]), cal.last_session)
    ]
    later = [d for d in future if d > session]
    next_open = cal.session_open(pd.Timestamp(later[0])).to_pydatetime() if later else None
    if too_late(now, next_open):
        # A következő tőzsdenap már kinyitott: a becslés ablakának egy része
        # lezajlott, a lenyomat utólagos válogatásnak látszana (docs/elo-futas.md).
        # A nap kiesett napként látszik, nem pótoljuk.
        log.warning("forecast_too_late", session=str(session), next_open=str(next_open))
        return {"session": str(session), "status": "too_late"}

    models = load_models(storage)
    frame = build_forecasts(features, prices, session, models, future, now.astimezone(UTC))
    if partial_package(frame, len(universe)):
        # A forrás az univerzum töredékére adott használható sort: nem mentjük,
        # hogy egy későbbi futás (a nyitás előtt) a teljes napot lenyomatolhassa.
        log.warning(
            "forecast_partial", session=str(session), instruments=int(frame["instrument_id"].nunique())
        )
        return {"session": str(session), "status": "partial"}
    if implied_allowed(now, next_open):
        frame = attach_implied(frame, implied_baseline(universe, prices, macro, session, future))
    else:
        log.info("implied_skipped", reason="a következő tőzsdenap már kinyitott", session=str(session))
        frame = attach_implied(frame, pd.DataFrame(), missing="stale_session")

    # Az ablakba eső ütemezett események (docs/naptar.md): a becsléssel
    # együtt, a lenyomat alá kerülnek, hogy később mérhető legyen, rosszabb-e
    # a modell az események közelében.
    from pipeline.events.calendar import attach_calendar

    frame = attach_calendar(frame)
    frame = attach_shock_flags(frame, prices, actions, universe, macro, session)
    package = _package_bytes(frame)
    entry = manifest_entry(session, package, frame, now.astimezone(UTC), len(universe))
    if dry_run:
        log.info("forecast_dry_run", **entry)
        return {**entry, "status": "dry_run"}

    storage.upload(RAW_BUCKET, package_path(session), package, "application/octet-stream")
    storage.upload(
        RAW_BUCKET,
        f"runs/forecast/{session.isoformat()}.json",
        json.dumps(entry, indent=2).encode(),
        "application/json",
    )
    log.info("forecast_saved", **entry)
    result: dict[str, object] = {**entry, "status": "saved"}

    # A modell-aréna kihívói. Bármi baj van velük, a fő csomag már elmentődött,
    # és a napi futás ettől nem bukik el: a nap a kihívóknál „not available”.
    try:
        challengers = load_challengers(storage)
        arena = build_challenger_forecasts(features, session, challengers, frame, now.astimezone(UTC))
        if arena.empty:
            result["arena"] = {"status": "not_available", "reason": "no_trained_challengers"}
        else:
            arena_package = _package_bytes(arena)
            arena_entry = manifest_entry(session, arena_package, arena, now.astimezone(UTC), len(universe))
            arena_entry["model_family"] = "arena"
            arena_entry["model_version"] = "+".join(
                f"{f} {v}"
                for f, v in sorted(set(zip(arena["model_family"], arena["model_version"], strict=True)))
            )
            storage.upload(RAW_BUCKET, arena_package_path(session), arena_package, "application/octet-stream")
            result["arena"] = {**arena_entry, "status": "saved"}
            log.info("arena_forecast_saved", forecasts=len(arena), families=arena["model_family"].nunique())
    except Exception as error:  # noqa: BLE001 — a fő becslés fontosabb
        log.warning("arena_forecast_failed", error=type(error).__name__)
        result["arena"] = {"status": "not_available", "reason": type(error).__name__}
    return result


def _package_bytes(frame: pd.DataFrame) -> bytes:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(frame, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calyze napi becslés")
    parser.add_argument("--local", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-dir", type=Path, help="ide írja a manifestet (a publikus repóban)")
    args = parser.parse_args(argv)

    logging_setup.configure()
    settings = load_settings()
    if args.local:
        storage: Storage = LocalStorage(args.local)
    elif settings.supabase_url and settings.supabase_secret_key:
        storage = SupabaseStorage(settings.supabase_url, settings.supabase_secret_key)
    else:
        log.error("storage_not_configured")
        return 2

    result = run(storage, datetime.now(UTC), dry_run=args.dry_run)
    if args.manifest_dir and result.get("status") == "saved":
        session = date.fromisoformat(str(result["session"]))
        out = args.manifest_dir / str(session.year)
        out.mkdir(parents=True, exist_ok=True)
        main_entry = {k: v for k, v in result.items() if k != "arena"}
        (out / f"{session.isoformat()}.json").write_text(json.dumps(main_entry, indent=2) + "\n")
        # A kihívók lenyomata külön mappába: a fő manifesteket olvasó kódok
        # (állapotoldal, időgép) így nem keverik a kettőt.
        arena = result.get("arena")
        if isinstance(arena, dict) and arena.get("status") == "saved":
            arena_dir = args.manifest_dir.parent / "manifests-arena" / str(session.year)
            arena_dir.mkdir(parents=True, exist_ok=True)
            (arena_dir / f"{session.isoformat()}.json").write_text(json.dumps(arena, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
