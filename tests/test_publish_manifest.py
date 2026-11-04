"""Feltöltés csak változáskor (J3): azonos tartalom → nincs írás; eltérő → van."""

from __future__ import annotations

import json

import pytest

from pipeline.ingest.storage import LocalStorage
from pipeline.publish.manifest import ChangedOnly


class Recording(LocalStorage):
    def __init__(self, root, fail_on: str | None = None) -> None:
        super().__init__(root)
        self.writes: list[str] = []
        self.fail_on = fail_on

    def upload(self, bucket, path, data, content_type):
        if path == self.fail_on:
            raise RuntimeError("szimulált hiba")
        self.writes.append(path)
        super().upload(bucket, path, data, content_type)


def publish(storage, files):
    up = ChangedOnly(storage, "display", ".manifest.json")
    try:
        for path, blob in files:
            up.put(path, blob, "application/json")
    finally:
        up.finish()
    return up


def test_csendes_futas_csak_a_manifesztet_irja(tmp_path):
    s = Recording(tmp_path)
    files = [(f"instruments/CZ{i:05d}.json", b"{}") for i in range(50)]
    publish(s, files)
    assert len(s.writes) == 51  # 50 fájl + a manifeszt
    s.writes.clear()
    up = publish(s, files)
    assert s.writes == [".manifest.json"]
    assert up.skipped == 50


def test_csak_az_eltero_fajl_megy_fel(tmp_path):
    s = Recording(tmp_path)
    publish(s, [("a.json", b"1"), ("b.json", b"1")])
    s.writes.clear()
    publish(s, [("a.json", b"1"), ("b.json", b"2"), ("c.json", b"3")])
    assert s.writes == ["b.json", "c.json", ".manifest.json"]


def test_elbukott_feltoltes_kiesik_a_manifesztbol(tmp_path):
    s = Recording(tmp_path)
    publish(s, [("a.json", b"1"), ("b.json", b"1")])
    failing = Recording(tmp_path, fail_on="b.json")
    with pytest.raises(RuntimeError):
        publish(failing, [("a.json", b"2"), ("b.json", b"2")])
    manifest = json.loads((tmp_path / "display" / ".manifest.json").read_bytes())
    assert "a.json" in manifest
    assert "b.json" not in manifest  # a következő futás újrapróbálja
    retry = Recording(tmp_path)
    publish(retry, [("a.json", b"2"), ("b.json", b"2")])
    assert retry.writes == ["b.json", ".manifest.json"]
