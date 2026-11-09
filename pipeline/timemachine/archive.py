"""Az időgép archívuma (spec/02 F9; spec/03 3.7; spec/05 #6).

Naponként egy fájl, az összes papírral: amit a Calyze azon a napon mondott.

**A forrás a lementett becslés-csomag, nem újraszámolás.** A csomag a
megjelenítés előtt mentődött, a lenyomata pedig a nyilvános repóba került.
Az archívum építése előtt a lenyomatot ÚJRASZÁMOLJUK, és összevetjük a
commitolttal. Ha nem egyezik, az archívum nem készül el: egy időgép, ami egy
ellenőrizetlen lenyomatot mutat, rosszabb, mint ha nem lenne.

A commit-link a manifesztet LÉTREHOZÓ commitra mutat, nem a legutolsóra. A
lenyomat létezését az első commit bizonyítja — egy későbbi szerkesztés (pl.
egy jelölés hozzáadása) nem lehet a bizonyíték.

Egy megírt archívum nem íródik felül: az a nap már elmúlt. Kivétel csak a
`--rebuild` kapcsolóval, és az a naplóba kerül.

Futtatás:
    uv run python -m pipeline.timemachine.archive
    uv run python -m pipeline.timemachine.archive --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from pipeline import log as logging_setup
from pipeline.config import RAW_BUCKET, load_settings
from pipeline.forecast.run import package_path
from pipeline.ingest.storage import LocalStorage, Storage, SupabaseStorage
from pipeline.publish.run import DISPLAY_BUCKET, format_forecast
from pipeline.universe import load_universe

log = logging_setup.get_logger(__name__)

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "manifests"
#: A kripto saját lenyomatai és archívuma (5. fázis, E4).
CRYPTO_MANIFESTS = REPO / "manifests-crypto"
CRYPTO_ARCHIVE_PREFIX = "archive-crypto"
#: A deviza saját lenyomatai és archívuma (6. fázis, F4).
FX_MANIFESTS = REPO / "manifests-fx"
FX_ARCHIVE_PREFIX = "archive-fx"
#: A kötvény saját lenyomatai és archívuma (7. fázis).
BONDS_MANIFESTS = REPO / "manifests-bonds"
BONDS_ARCHIVE_PREFIX = "archive-bonds"
PUBLIC_REPO_URL = "https://github.com/sigmundsabo/calyze-data"
ARCHIVE_PREFIX = "archive"
INDEX_PATH = f"{ARCHIVE_PREFIX}/index.json"

#: Ez alatt a lefedettség alatt a nap részleges (ugyanaz a küszöb, mint a
#: becslésnél és a publikus naplóban).
MIN_SESSION_COVERAGE = 0.8


class IntegrityError(RuntimeError):
    """A lementett csomag lenyomata nem egyezik a nyilvánosan commitolttal."""


def verify_package(blob: bytes, manifest: dict[str, object]) -> str:
    """A csomag lenyomata egyezik-e a manifesztben commitolttal.

    Nem elég kiírni a lenyomatot: ha nem vetjük össze semmivel, akkor csak egy
    szám, ami hitelesnek látszik.
    """
    digest = hashlib.sha256(blob).hexdigest()
    expected = str(manifest.get("sha256", ""))
    if digest != expected:
        raise IntegrityError(
            f"A {manifest.get('session')} napi csomag lenyomata nem egyezik: "
            f"a tárban {digest[:16]}…, a manifesztben {expected[:16]}…"
        )
    return digest


def adding_commit(path: Path, repo: Path = REPO) -> dict[str, str] | None:
    """A commit, amelyik a fájlt LÉTREHOZTA — nem a legutolsó, ami módosította.

    `None`, ha a git-történet nem elérhető (sekély klón): olyankor a felület
    nem mutat commit-linket, inkább mint hogy rosszat mutasson.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        # Sekély klónban a csonkolt első commit minden fájlt „létrehoz”: a
        # keresés a legutolsó commitot adná vissza. Akkor inkább nincs link.
        shallow = subprocess.run(  # noqa: S603
            [git, "rev-parse", "--is-shallow-repository"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if shallow == "true":
            return None
        # A paraméterek rögzített lista, shell nélkül; az útvonal a saját
        # manifeszt-mappánkból jön, nem felhasználói bemenetből.
        out = subprocess.run(  # noqa: S603
            [git, "log", "--diff-filter=A", "--format=%H%x09%cI", "--", str(path.relative_to(repo))],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None
    if not out:
        return None
    # Ha a fájlt többször is létrehozták (törlés után újra), a legelső számít.
    sha, when = out.splitlines()[-1].split("\t")
    return {"sha": sha, "time": when, "url": f"{PUBLIC_REPO_URL}/commit/{sha}"}


def build_archive(
    manifest: dict[str, object],
    commit: dict[str, str] | None,
    package: pd.DataFrame,
    universe: pd.DataFrame,
    archived_at: datetime,
) -> dict[str, object]:
    """Egy nap archívuma: minden papír becslése, ahogy aznap lementődött."""
    names = universe.set_index("instrument_id")[["ticker", "name"]].to_dict("index")
    instruments: dict[str, object] = {}
    for instrument_id, group in package.groupby("instrument_id", sort=True):
        meta = names.get(instrument_id, {})
        rows = group.sort_values("horizon").to_dict("records")
        instruments[str(instrument_id)] = {
            "ticker": meta.get("ticker", str(instrument_id)),
            "name": meta.get("name", ""),
            "close": float(rows[0]["close"]) if rows and pd.notna(rows[0].get("close")) else None,
            "forecasts": [format_forecast(r) for r in rows],
        }

    # A rezsim a csomagban van, soronként: aznap a becsléssel együtt mentődött,
    # nem egy később számolt táblából jön.
    regime = None
    if "regime" in package and not package["regime"].dropna().empty:
        regime = str(package["regime"].dropna().mode().iloc[0])

    covered = len(instruments)
    universe_size = int(manifest.get("universe") or 0)  # type: ignore[call-overload]
    return {
        "session": manifest["session"],
        "made_at": manifest.get("made_at"),
        "archived_at": archived_at.astimezone(UTC).replace(microsecond=0).isoformat(),
        "sha256": manifest["sha256"],
        "commit": commit,
        "model": f"{manifest.get('model_family', '?')} {manifest.get('model_version', '')}".strip(),
        "regime": regime,
        "instruments_with_forecast": covered,
        "universe": universe_size or None,
        "partial": bool(universe_size) and covered < universe_size * MIN_SESSION_COVERAGE,
        "instruments": instruments,
    }


def load_manifests(root: Path = MANIFESTS) -> list[tuple[Path, dict[str, object]]]:
    items = []
    for path in sorted(root.glob("*/*.json")):
        items.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return items


def published_summary(storage: Storage, path: str = "latest.json") -> dict[str, object] | None:
    """A most kint lévő napi összefoglaló, ha van."""
    blob = storage.download(DISPLAY_BUCKET, path)
    return None if blob is None else json.loads(blob)


def run(
    storage: Storage, now: datetime, dry_run: bool = False, rebuild: bool = False, kind: str = "equity"
) -> dict[str, object]:
    crypto = kind == "crypto"
    if kind == "bonds":
        from pipeline.bonds.forecast import package_path as bonds_package_path
        from pipeline.universe import load_bonds_universe

        universe = load_bonds_universe()
        manifests_root, prefix, path_of = BONDS_MANIFESTS, BONDS_ARCHIVE_PREFIX, bonds_package_path
    elif kind == "fx":
        from pipeline.fx.forecast import package_path as fx_package_path
        from pipeline.universe import load_fx_universe

        universe = load_fx_universe()
        manifests_root, prefix, path_of = FX_MANIFESTS, FX_ARCHIVE_PREFIX, fx_package_path
    elif crypto:
        from pipeline.crypto.forecast import package_path as crypto_package_path
        from pipeline.universe import load_crypto_universe

        universe = load_crypto_universe()
        manifests_root, prefix, path_of = CRYPTO_MANIFESTS, CRYPTO_ARCHIVE_PREFIX, crypto_package_path
    else:
        universe = load_universe()
        manifests_root, prefix, path_of = MANIFESTS, ARCHIVE_PREFIX, package_path
    # A verdict nincs a becslés-csomagban: az a mérésből jön. Csak akkor
    # archiválhatjuk, ha a most kint lévő összefoglaló ugyanarra a napra szól —
    # egy korábbi napra a mai verdictet írni utólagos bölcsesség lenne. A
    # kriptónál ez a kripto-összefoglaló.
    summary_now = published_summary(
        storage, f"{kind}/latest.json" if kind in ("crypto", "fx", "bonds") else "latest.json"
    )
    index: list[dict[str, object]] = []
    written = kept = 0

    for path, manifest in load_manifests(manifests_root):
        session = str(manifest["session"])
        target = f"{prefix}/{session}.json"
        commit = adding_commit(path)

        existing = None if rebuild else storage.download(DISPLAY_BUCKET, target)
        if existing is not None:
            archive = json.loads(existing)
            kept += 1
        else:
            blob = storage.download(RAW_BUCKET, path_of(date.fromisoformat(session)))
            if blob is None:
                log.warning("archive_package_missing", session=session)
                continue
            verify_package(blob, manifest)
            package = pd.read_parquet(io.BytesIO(blob))
            archive = build_archive(manifest, commit, package, universe, now)
            same_day = summary_now is not None and str(summary_now.get("session")) == session
            archive["verdict_archived"] = same_day
            archive["verdict"] = summary_now.get("verdict") if same_day and summary_now else None
            if not dry_run:
                storage.upload(
                    DISPLAY_BUCKET,
                    target,
                    json.dumps(archive, ensure_ascii=False, separators=(",", ":"), default=str).encode(),
                    "application/json",
                )
            written += 1
            if rebuild:
                log.info("archive_rebuilt", session=session)

        index.append(
            {
                "session": session,
                "made_at": archive.get("made_at"),
                "sha256": archive.get("sha256"),
                "commit": archive.get("commit"),
                "instruments_with_forecast": archive.get("instruments_with_forecast"),
                "partial": archive.get("partial"),
            }
        )

    index.sort(key=lambda e: str(e["session"]), reverse=True)
    if not dry_run:
        storage.upload(
            DISPLAY_BUCKET,
            f"{prefix}/index.json",
            json.dumps({"sessions": index}, separators=(",", ":"), default=str).encode(),
            "application/json",
        )
    summary = {"sessions": len(index), "written": written, "kept": kept, "dry_run": dry_run}
    log.info("archive_done", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Az időgép archívuma")
    parser.add_argument("--local", type=Path, default=None, help="helyi tár a Supabase helyett")
    parser.add_argument("--dry-run", action="store_true", help="ellenőriz és épít, de nem ír")
    parser.add_argument("--rebuild", action="store_true", help="a meglévő archívumokat is újraírja")
    parser.add_argument(
        "--kind", choices=["equity", "crypto", "fx", "bonds"], default="equity", help="melyik eszközosztály"
    )
    args = parser.parse_args()

    logging_setup.configure()
    storage: Storage
    if args.local is not None:
        storage = LocalStorage(args.local)
    else:
        settings = load_settings()
        storage = SupabaseStorage(settings.supabase_url or "", settings.supabase_secret_key or "")
    result = run(storage, datetime.now(UTC), dry_run=args.dry_run, rebuild=args.rebuild, kind=args.kind)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
