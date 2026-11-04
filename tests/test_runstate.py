"""A lépés-kapu (J2): új adat nélkül a nehéz lépés nem fut; a hibás próba pótlódik."""

from __future__ import annotations

import pytest

from pipeline.ingest.storage import LocalStorage
from pipeline.runstate import gated, record_change


def test_helyben_mindig_fut(tmp_path, monkeypatch):
    monkeypatch.delenv("CALYZE_CHANGES_FILE", raising=False)
    s = LocalStorage(tmp_path)
    assert gated(s, "publish-equity", lambda: "kesz") == "kesz"
    assert gated(s, "publish-equity", lambda: "ujra") == "ujra"


def test_valtozas_nelkul_a_sikeres_lepes_nem_fut_ujra(tmp_path, monkeypatch):
    monkeypatch.setenv("CALYZE_CHANGES_FILE", str(tmp_path / "changes.txt"))
    s = LocalStorage(tmp_path / "store")
    assert gated(s, "resolve-equity", lambda: "elso") == "elso"  # még nincs állapot: fut
    assert gated(s, "resolve-equity", lambda: "masodik") is None  # nincs új adat: kimarad


def test_uj_adat_utan_fut(tmp_path, monkeypatch):
    monkeypatch.setenv("CALYZE_CHANGES_FILE", str(tmp_path / "changes.txt"))
    s = LocalStorage(tmp_path / "store")
    gated(s, "publish-equity", lambda: 1)
    record_change("market-data-raw", "prices_daily/year=2026.parquet")
    assert gated(s, "publish-equity", lambda: 2) == 2


def test_a_naplo_es_a_manifeszt_nem_szamit_valtozasnak(tmp_path, monkeypatch):
    monkeypatch.setenv("CALYZE_CHANGES_FILE", str(tmp_path / "changes.txt"))
    record_change("market-data-raw", "runs/ingest/2026-11-04.json")
    record_change("display", ".manifest.json")
    assert not (tmp_path / "changes.txt").exists()


def test_az_elbukott_lepes_a_kovetkezo_futasban_potlodik(tmp_path, monkeypatch):
    monkeypatch.setenv("CALYZE_CHANGES_FILE", str(tmp_path / "changes.txt"))
    s = LocalStorage(tmp_path / "store")

    def boom() -> None:
        raise RuntimeError("hiba")

    with pytest.raises(RuntimeError):
        gated(s, "publish-fx", boom)
    assert gated(s, "publish-fx", lambda: "potolva") == "potolva"
