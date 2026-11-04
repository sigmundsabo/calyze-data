import pytest

from pipeline.ingest.storage import StorageError, SupabaseStorage


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class Session:
    def __init__(self, get_resp, post_resp=None):
        self.headers = {}
        self.get_resp = get_resp
        self.post_resp = post_resp or Resp(200, {})
        self.posted = []

    def get(self, url, timeout):
        return self.get_resp

    def post(self, url, json=None, timeout=None, **_):
        self.posted.append((url, json))
        return self.post_resp


def make(get_resp, post_resp=None):
    s = SupabaseStorage("https://example.supabase.co", "fake-key-for-tests")
    s.session = Session(get_resp, post_resp)
    return s


def test_public_bucket_stops_the_run():
    # A nyers adat licenc miatt nem lehet nyilvános (spec/05, 2.4).
    with pytest.raises(StorageError, match="NYILVÁNOS"):
        make(Resp(200, {"id": "market-data-raw", "public": True})).ensure_private_bucket("market-data-raw")


def test_missing_bucket_is_created_private():
    s = make(Resp(400, {"statusCode": "404", "error": "Bucket not found"}))
    s.ensure_private_bucket("market-data-raw")
    url, body = s.session.posted[0]
    assert url.endswith("/storage/v1/bucket")
    assert body == {"id": "market-data-raw", "name": "market-data-raw", "public": False}


def test_error_message_never_contains_the_response_body():
    s = make(Resp(500, {"secret": "should-not-leak"}))
    with pytest.raises(StorageError) as err:
        s.ensure_private_bucket("x")
    assert "should-not-leak" not in str(err.value)


def test_az_atmeneti_hibat_ujraprobalja(monkeypatch) -> None:
    """Egy négyórás futás nem dőlhet el egyetlen 502-n."""
    import time as time_module

    from pipeline.ingest.storage import SupabaseStorage

    storage = SupabaseStorage("https://example.invalid", "sb_secret_x")
    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)

    calls: list[str] = []

    class Response:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.content = b"adat"
            self.text = ""

    def fake_request(_method: str, _url: str, **kwargs: object) -> Response:
        # A timeout egyszer és csak egyszer kerül be — a feltöltés korábban
        # maga is átadta, és a kérés „multiple values for timeout”-tal halt el.
        assert "timeout" in kwargs
        calls.append(_url)
        return Response(502 if len(calls) < 3 else 200)

    monkeypatch.setattr(storage.session, "request", fake_request)
    assert storage.download("bucket", "path.parquet") == b"adat"
    assert len(calls) == 3, "a harmadik próbálkozás sikerült"


def test_a_vegleges_hibat_nem_probalja_ujra(monkeypatch) -> None:
    """A 4xx-en az újrapróbálás nem segít, csak késlelteti a hibát."""
    from pipeline.ingest.storage import StorageError, SupabaseStorage

    storage = SupabaseStorage("https://example.invalid", "sb_secret_x")
    calls: list[str] = []

    class Response:
        status_code = 403
        content = b""
        text = ""

    def fake_request(_method: str, _url: str, **_kwargs: object) -> Response:
        calls.append(_url)
        return Response()

    monkeypatch.setattr(storage.session, "request", fake_request)
    try:
        storage.download("bucket", "path.parquet")
    except StorageError:
        pass
    assert len(calls) == 1


def test_az_idotullepest_is_ujraprobalja(monkeypatch) -> None:
    """A minta-aréna egy lassú feltöltési válaszon veszett el: ez nem ismétlődhet."""
    import time as time_module

    import requests

    from pipeline.ingest.storage import SupabaseStorage

    storage = SupabaseStorage("https://example.invalid", "sb_secret_x")
    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)
    calls: list[str] = []

    class Response:
        status_code = 200
        content = b"adat"
        text = ""

    def flaky(_method: str, url: str, **_kwargs: object) -> Response:
        calls.append(url)
        if len(calls) < 3:
            raise requests.exceptions.ReadTimeout("lassú válasz")
        return Response()

    monkeypatch.setattr(storage.session, "request", flaky)
    assert storage.download("bucket", "path.parquet") == b"adat"
    assert len(calls) == 3


def test_a_tartos_idotullepes_vegul_hibat_dob(monkeypatch) -> None:
    """Ha minden próbálkozás elhal, a hiba nem nyelődik el."""
    import time as time_module

    import pytest
    import requests

    from pipeline.ingest.storage import SupabaseStorage

    storage = SupabaseStorage("https://example.invalid", "sb_secret_x")
    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)

    def dead(_method: str, _url: str, **_kwargs: object) -> None:
        raise requests.exceptions.ConnectTimeout("nincs kapcsolat")

    monkeypatch.setattr(storage.session, "request", dead)
    with pytest.raises(requests.exceptions.ConnectTimeout):
        storage.download("bucket", "path.parquet")


class Counting:
    """A `_request` kéréseit számolja; a letöltés mindig ugyanazt adja."""

    def __init__(self, body: bytes = b"abc") -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []
        self.body = body

    def request(self, method, url, **_):
        self.calls.append(method)
        r = Resp(200, {})
        r.content = self.body
        return r


def test_a_futason_beluli_gyorsitotar_egyszer_tolt_le(tmp_path):
    # J1: a job lépései ugyanazt a partíciót olvassák; a második már a lemezről jön.
    s = SupabaseStorage("https://example.supabase.co", "fake-key-for-tests", cache_dir=tmp_path)
    s.session = Counting()
    assert s.download("market-data-raw", "prices/year=2026.parquet") == b"abc"
    assert s.download("market-data-raw", "prices/year=2026.parquet") == b"abc"
    assert s.session.calls == ["GET"]
    assert s.traffic.downloads == 1
    assert s.traffic.cache_hits == 1


def test_azonos_tartalmat_nem_ir_ujra_az_eltérőt_igen(tmp_path):
    s = SupabaseStorage("https://example.supabase.co", "fake-key-for-tests", cache_dir=tmp_path)
    s.session = Counting()
    s.download("market-data-raw", "p.parquet")
    s.upload("market-data-raw", "p.parquet", b"abc", "application/octet-stream")
    assert s.session.calls == ["GET"]  # azonos: nincs feltöltés
    s.upload("market-data-raw", "p.parquet", b"abcd", "application/octet-stream")
    assert s.session.calls == ["GET", "POST"]
    assert s.traffic.uploads_skipped == 1
    assert s.traffic.uploads == 1


def test_gyorsitotar_nelkul_minden_letoltes_kimegy(tmp_path, monkeypatch):
    monkeypatch.delenv("CALYZE_STORAGE_CACHE", raising=False)
    s = SupabaseStorage("https://example.supabase.co", "fake-key-for-tests")
    s.session = Counting()
    s.download("b", "x")
    s.download("b", "x")
    assert s.session.calls == ["GET", "GET"]


def test_az_ismetelt_letoltes_nem_irja_at_az_azonos_sort():
    import pandas as pd

    from pipeline.ingest.partitions import upsert

    old = pd.DataFrame(
        {
            "instrument_id": ["A"],
            "date": [pd.Timestamp("2026-11-03").date()],
            "close": [10.0],
            "fetched_at": ["t1"],
        }
    )
    same = old.assign(fetched_at="t2")
    assert upsert(old, same)["fetched_at"].tolist() == ["t1"]  # csak az időbélyeg más: a régi marad
    changed = old.assign(close=10.5, fetched_at="t2")
    out = upsert(old, changed)
    assert out["close"].tolist() == [10.5]
    assert out["fetched_at"].tolist() == ["t2"]
