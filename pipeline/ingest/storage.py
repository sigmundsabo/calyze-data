"""Privát tár a nyers árfolyamnak (spec/05, 1. fejezet és 2.4).

A nyers adat licenc miatt nem kerülhet a publikus repóba, ezért egy **privát**
Supabase Storage-tárolóban él, amit csak a pipeline ér el a titkos kulccsal.
Ha a tároló valaha nyilvánosra állna, a futás megáll — inkább ne fusson, mint
hogy kiszolgálja a nyers adatot.

A helyi változat (`LocalStorage`) próbafuttatáshoz és teszthez való.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import time
from pathlib import Path
from typing import Protocol

import requests

from pipeline.log import get_logger

log = get_logger(__name__)


class StorageError(RuntimeError):
    pass


class Storage(Protocol):
    def ensure_private_bucket(self, bucket: str) -> None: ...
    def download(self, bucket: str, path: str) -> bytes | None: ...
    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None: ...
    def list(self, bucket: str, prefix: str) -> list[str]: ...


class SupabaseStorage:
    """A Supabase Storage REST API-ja, az új `sb_secret_` kulccsal.

    A kulcs az `apikey` és az `Authorization` fejlécbe is kerül; az átjáró a
    `Bearer sb_…` értéket a szerepnek megfelelő belső tokenre cseréli.
    """

    #: Átmeneti szerverhibák, amiket érdemes újrapróbálni. A 4xx nincs köztük:
    #: azon az újrapróbálás nem segít, csak késlelteti a hibát.
    RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
    RETRIES = 4

    def __init__(self, url: str, secret_key: str, timeout: float = 60, cache_dir: Path | None = None) -> None:
        self.base = f"{url.rstrip('/')}/storage/v1"
        self.session = requests.Session()
        self.session.headers.update({"apikey": secret_key, "Authorization": f"Bearer {secret_key}"})
        self.timeout = timeout
        # Futáson belüli lemez-gyorsítótár (`docs/terv-egress.md`, J1): a job
        # lépései (mutatók, becslés, kiértékelés, megjelenítés) ugyanazt az
        # árfolyamot olvassák; így egyszer jön le. Csak az ideiglenes gépen él:
        # futások között nincs cache, mert a publikus repó cache-éből egy idegen
        # pull request a licencelt nyers adatot is visszaolvashatná.
        env = os.environ.get("CALYZE_STORAGE_CACHE")
        self.cache_dir = cache_dir if cache_dir is not None else (Path(env) if env else None)
        self.traffic = Traffic()
        atexit.register(self.traffic.report)

    def _cached(self, bucket: str, path: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / bucket / path

    def _request(self, method: str, url: str, **kwargs: object) -> requests.Response:
        """Kérés újrapróbálással az átmeneti hibákra.

        Kétféle átmeneti hiba van, és mindkettőt újrapróbáljuk:
          - a szerver válaszol, de hibakóddal (429, 5xx),
          - a kapcsolat maga szakad meg vagy nem jön válasz időben.

        Az első változat csak az elsőt kezelte, és a minta-aréna egy 10 perces
        számítás után, egyetlen lassú feltöltési válaszon veszett el.

        A várakozás duplázódik (1, 2, 4 mp), hogy egy terhelt kiszolgálót ne
        nyomjunk tovább. Az utolsó próbálkozás hibája már nem nyelődik el.
        """
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(self.RETRIES):
            last_try = attempt == self.RETRIES - 1
            try:
                response = self.session.request(method, url, **kwargs)  # type: ignore[arg-type]
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if last_try:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code not in self.RETRY_STATUS or last_try:
                return response
            time.sleep(2**attempt)
        raise AssertionError("elérhetetlen: a ciklus mindig visszatér vagy kivételt dob")  # pragma: no cover

    def _check(self, response: requests.Response, what: str) -> None:
        if response.status_code >= 300:
            # A válasz törzsét nem írjuk ki: nem tudjuk, mit tartalmaz.
            raise StorageError(f"{what}: HTTP {response.status_code}")

    @staticmethod
    def _not_found(response: requests.Response) -> bool:
        if response.status_code == 404:
            return True
        if response.status_code == 400:
            try:
                body = response.json()
            except ValueError:
                return False
            return str(body.get("statusCode")) == "404" or body.get("error") in {
                "not_found",
                "Bucket not found",
            }
        return False

    def ensure_private_bucket(self, bucket: str) -> None:
        r = self.session.get(f"{self.base}/bucket/{bucket}", timeout=self.timeout)
        if self._not_found(r):
            created = self.session.post(
                f"{self.base}/bucket",
                json={"id": bucket, "name": bucket, "public": False},
                timeout=self.timeout,
            )
            self._check(created, "tároló létrehozása")
            log.info("bucket_created", bucket=bucket)
            return
        self._check(r, "tároló lekérése")
        if r.json().get("public") is not False:
            raise StorageError(
                f"A(z) {bucket} tároló NYILVÁNOS. A nyers adat nem lehet publikus — a futás megáll."
            )

    def download(self, bucket: str, path: str) -> bytes | None:
        local = self._cached(bucket, path)
        if local is not None and local.exists():
            self.traffic.cache_hits += 1
            return local.read_bytes()
        r = self._request("GET", f"{self.base}/object/{bucket}/{path}")
        if self._not_found(r):
            return None
        self._check(r, f"letöltés ({path})")
        self.traffic.downloads += 1
        self.traffic.download_bytes += len(r.content)
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(r.content)
        return r.content

    #: A Supabase ingyenes sávján a feltöltés fájlonkénti felső határa.
    MAX_UPLOAD_BYTES = 50 * 1024 * 1024

    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        if len(data) > self.MAX_UPLOAD_BYTES:
            raise StorageError(
                f"feltöltés ({path}): {len(data) / 1048576:.0f} MB, a határ "
                f"{self.MAX_UPLOAD_BYTES // 1048576} MB — bontsd kisebb fájlokra vagy összesíts"
            )
        local = self._cached(bucket, path)
        # Ha ebben a futásban már letöltöttük, és bitre azonos, nem írjuk újra:
        # a napi újrapróbák így nem írják felül ugyanazt a partíciót.
        if local is not None and local.exists() and _digest(local.read_bytes()) == _digest(data):
            self.traffic.uploads_skipped += 1
            return
        r = self._request(
            "POST",
            f"{self.base}/object/{bucket}/{path}",
            data=data,
            headers={"Content-Type": content_type, "x-upsert": "true", "cache-control": "no-store"},
        )
        self._check(r, f"feltöltés ({path})")
        from pipeline.runstate import record_change

        record_change(bucket, path)
        self.traffic.uploads += 1
        self.traffic.upload_bytes += len(data)
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)

    def list(self, bucket: str, prefix: str) -> list[str]:
        r = self.session.post(
            f"{self.base}/object/list/{bucket}",
            json={"prefix": prefix, "limit": 1000, "sortBy": {"column": "name", "order": "asc"}},
            timeout=self.timeout,
        )
        self._check(r, f"listázás ({prefix})")
        return [f"{prefix.rstrip('/')}/{item['name']}" for item in r.json() if item.get("id")]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Traffic:
    """A tároló-forgalom számlálói egy folyamatra (J7). A naplóba csak szám kerül."""

    def __init__(self) -> None:
        self.downloads = 0
        self.download_bytes = 0
        self.cache_hits = 0
        self.uploads = 0
        self.upload_bytes = 0
        self.uploads_skipped = 0

    def report(self) -> None:
        if self.downloads or self.uploads or self.cache_hits or self.uploads_skipped:
            log.info(
                "storage_traffic",
                downloads=self.downloads,
                download_mb=round(self.download_bytes / 1048576, 1),
                cache_hits=self.cache_hits,
                uploads=self.uploads,
                upload_mb=round(self.upload_bytes / 1048576, 1),
                uploads_skipped=self.uploads_skipped,
            )


class LocalStorage:
    """Helyi mappa a próbafuttatáshoz. A `data/` a .gitignore-ban van."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def ensure_private_bucket(self, bucket: str) -> None:
        (self.root / bucket).mkdir(parents=True, exist_ok=True)

    def download(self, bucket: str, path: str) -> bytes | None:
        p = self.root / bucket / path
        return p.read_bytes() if p.exists() else None

    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        p = self.root / bucket / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list(self, bucket: str, prefix: str) -> list[str]:
        base = self.root / bucket / prefix
        if not base.exists():
            return []
        return sorted(f"{prefix.rstrip('/')}/{p.name}" for p in base.iterdir() if p.is_file())
