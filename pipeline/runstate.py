"""Mikor kell egy lépésnek futnia (a privát repó `docs/terv-egress.md`, J2).

A napi futás naponta többször indul (újrapróbák). A kiértékelés és a
megjelenítés évekre visszamenő árfolyamot tölt le; ha a futás korábbi lépései
semmi újat nem írtak a tárolóba, nincs mit újraszámolniuk.

Két jel dönt:

1. **Változott-e valami ebben a futásban:** a tároló minden valódi
   feltöltése (a `runs/` naplón kívül) egy sort ír a `CALYZE_CHANGES_FILE`
   fájlba. A fájl a GitHub-futás ideiglenes lemezén él.
2. **Sikerült-e a lépés legutóbb:** a `runs/state/<lépés>.json` a privát
   tárban. Ha a legutóbbi próba elbukott, a következő futás akkor is lefut.

Helyben és tesztben (nincs `CALYZE_CHANGES_FILE`) a lépés mindig fut.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import RAW_BUCKET
from pipeline.ingest.storage import Storage
from pipeline.log import get_logger

log = get_logger(__name__)

STATE_PREFIX = "runs/state"


def changes_file() -> Path | None:
    value = os.environ.get("CALYZE_CHANGES_FILE")
    return Path(value) if value else None


def record_change(bucket: str, path: str) -> None:
    """A tároló hívja minden valódi feltöltés után; a futásnapló és az állapot nem számít."""
    target = changes_file()
    if target is None or path.startswith("runs/") or path.endswith(".manifest.json"):
        return
    with target.open("a", encoding="utf-8") as f:
        f.write(f"{bucket}/{path}\n")


def changed_in_this_run() -> bool:
    target = changes_file()
    return target is not None and target.exists() and target.stat().st_size > 0


def should_run(storage: Storage, step: str) -> bool:
    if changes_file() is None or changed_in_this_run():
        return True
    raw = storage.download(RAW_BUCKET, f"{STATE_PREFIX}/{step}.json")
    try:
        status = json.loads(raw).get("status") if raw else None
    except ValueError:
        status = None
    if status != "ok":
        return True
    log.info("step_skipped", step=step, reason="nothing_new")
    return False


def mark(storage: Storage, step: str, ok: bool) -> None:
    payload = {"status": "ok" if ok else "failed", "at": datetime.now(UTC).isoformat(timespec="seconds")}
    storage.upload(
        RAW_BUCKET, f"{STATE_PREFIX}/{step}.json", json.dumps(payload).encode(), "application/json"
    )


def gated[T](storage: Storage, step: str, work: Callable[[], T]) -> T | None:
    """A lépés csak akkor fut, ha kell; az eredményét (siker vagy hiba) feljegyzi."""
    if not should_run(storage, step):
        return None
    try:
        result = work()
    except BaseException:
        mark(storage, step, ok=False)
        raise
    mark(storage, step, ok=True)
    return result
