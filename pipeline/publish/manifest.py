"""Feltöltés csak változáskor (a privát repó `docs/terv-egress.md`, J3).

A megjelenítési fájlok zöme (papíronként a csomag és az idősor) a nap első
futása után már nem változik, a pipeline mégis naponta többször mindet
újraírta (kb. 3 200 írás). Egy `útvonal → sha256` manifeszt alapján csak az
új vagy eltérő fájl megy fel: egy csendes futás egy letöltés és egy feltöltés.

A manifeszt csak útvonalat és hasht tartalmaz (a publikus repó szabálya
szerint kulcs és felhasználói adat soha), és **csak a sikeres feltöltések
után** íródik: ha egy feltöltés elbukik, az útvonala kiesik belőle, így a
következő futás újrapróbálja. A két írás így nem csúszhat szét.
"""

from __future__ import annotations

import hashlib
import json

from pipeline.ingest.storage import Storage
from pipeline.log import get_logger

log = get_logger(__name__)


class ChangedOnly:
    def __init__(self, storage: Storage, bucket: str, manifest_path: str) -> None:
        self.storage = storage
        self.bucket = bucket
        self.manifest_path = manifest_path
        raw = storage.download(bucket, manifest_path)
        try:
            loaded = json.loads(raw) if raw else {}
        except ValueError:
            loaded = {}
        self.known: dict[str, str] = (
            {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}
        )
        self.uploaded = 0
        self.skipped = 0

    def put(self, path: str, blob: bytes, content_type: str) -> None:
        digest = hashlib.sha256(blob).hexdigest()
        if self.known.get(path) == digest:
            self.skipped += 1
            return
        # A régi hash a feltöltés ELŐTT kiesik: ha a feltöltés elbukik, a
        # következő futás biztosan újrapróbálja.
        self.known.pop(path, None)
        self.storage.upload(self.bucket, path, blob, content_type)
        self.known[path] = digest
        self.uploaded += 1

    def finish(self) -> None:
        """A manifeszt kiírása; a `put` hibája esetén is hívandó (`try/finally`)."""
        blob = json.dumps(self.known, sort_keys=True, separators=(",", ":")).encode()
        self.storage.upload(self.bucket, self.manifest_path, blob, "application/json")
        log.info("display_upload", uploaded=self.uploaded, skipped=self.skipped)
