"""Az időgép csak azt mutathatja, ami aznap lementődött — ellenőrzötten."""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from pipeline.timemachine.archive import (
    IntegrityError,
    adding_commit,
    build_archive,
    verify_package,
)


def package_frame() -> pd.DataFrame:
    rows = []
    for instrument in ("CZ00001", "CZ00002"):
        for horizon in (5, 20, 60):
            rows.append(
                {
                    "instrument_id": instrument,
                    "horizon": horizon,
                    "prob_up": 0.53,
                    "baseline_prob": 0.51,
                    "baseline_id": "naive",
                    "expected_return": 0.004,
                    "band_low": -0.04,
                    "band_high": 0.05,
                    "price_low": 96.0,
                    "price_high": 105.0,
                    "expected_price": 100.4,
                    "close": 100.0,
                    "made_at": "2026-09-24T13:14:07+00:00",
                    "target_session": "2026-10-21",
                    "regime": "normal",
                    "contributions": '[{"feature":"mom_20","value":0.01}]',
                }
            )
    return pd.DataFrame(rows)


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def manifest_for(blob: bytes, **extra: object) -> dict[str, object]:
    return {
        "session": "2026-09-23",
        "made_at": "2026-09-24T13:14:07+00:00",
        "model_family": "lgbm-core",
        "model_version": "v1",
        "sha256": hashlib.sha256(blob).hexdigest(),
        **extra,
    }


def test_az_egyezo_lenyomat_atmegy() -> None:
    blob = parquet_bytes(package_frame())
    assert verify_package(blob, manifest_for(blob)) == hashlib.sha256(blob).hexdigest()


def test_a_megvaltoztatott_csomagra_nem_keszul_archivum() -> None:
    """Egyetlen becslés átírása után a lenyomat nem egyezik, és az építés megáll."""
    original = package_frame()
    blob = parquet_bytes(original)
    manifest = manifest_for(blob)

    tampered = original.copy()
    tampered.loc[0, "prob_up"] = 0.99
    with pytest.raises(IntegrityError, match="nem egyezik"):
        verify_package(parquet_bytes(tampered), manifest)


def test_az_archivum_a_lementett_szamokat_hordozza() -> None:
    frame = package_frame()
    blob = parquet_bytes(frame)
    archive = build_archive(
        manifest_for(blob, universe=620),
        {"sha": "abc", "time": "2026-09-24T13:17:02Z", "url": "https://example/commit/abc"},
        frame,
        pd.DataFrame({"instrument_id": ["CZ00001"], "ticker": ["AAPL"], "name": ["Apple Inc."]}),
        datetime(2026, 9, 26, tzinfo=UTC),
    )
    first = archive["instruments"]["CZ00001"]  # type: ignore[index]
    assert first["ticker"] == "AAPL"
    assert [f["horizon"] for f in first["forecasts"]] == [5, 20, 60]
    assert first["forecasts"][0]["prob_up"] == 0.53
    # A rezsim a csomagból jön, nem egy később számolt táblából.
    assert archive["regime"] == "normal"
    # Két papír a 620-ból: ez részleges nap, és az archívum ki is mondja.
    assert archive["partial"] is True
    # A becslés és az archiválás ideje két külön mező: nem mossuk össze.
    assert archive["made_at"] != archive["archived_at"]


def test_a_commit_link_a_letrehozo_commitra_mutat() -> None:
    """A 09-22-i manifesztet utólag szerkesztettük — a link mégis az elsőre mutat."""
    repo = Path(__file__).resolve().parents[1]
    path = repo / "manifests" / "2026" / "2026-09-22.json"
    commit = adding_commit(path, repo)
    if commit is None:
        pytest.skip("sekély klón: nincs git-történet")
    import shutil
    import subprocess

    git = shutil.which("git")
    assert git is not None
    all_commits = subprocess.run(  # noqa: S603 — rögzített paraméterek, shell nélkül
        [git, "log", "--format=%H", "--", str(path.relative_to(repo))],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert len(all_commits) >= 2, "a fájlt létrehozása után módosítottuk"
    assert commit["sha"] == all_commits[-1], "a legelső commit, nem a legutolsó"
    assert commit["url"].endswith(commit["sha"])


def test_sekely_klonban_nincs_commit_link(tmp_path: Path) -> None:
    """A csonkolt történetben minden fájl „új”: rossz link helyett nincs link."""
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("nincs git")
    source = tmp_path / "source"
    source.mkdir()
    run = lambda *args, cwd=source: subprocess.run(  # noqa: E731, S603
        [git, *args], cwd=cwd, check=True, capture_output=True
    )
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (source / "m.json").write_text("{}")
    run("add", "m.json")
    run("commit", "-q", "-m", "létrehozás")
    (source / "m.json").write_text('{"x": 1}')
    run("commit", "-q", "-am", "módosítás")
    clone = tmp_path / "clone"
    run("clone", "-q", "--depth", "1", f"file://{source}", str(clone), cwd=tmp_path)
    assert adding_commit(clone / "m.json", clone) is None
    assert adding_commit(source / "m.json", source) is not None


def test_a_nap_fajl_karcsu_a_becslesek_papironkent_kulon() -> None:
    import json

    from pipeline.timemachine.archive import index_entry, split_archive

    frame = package_frame()
    blob = parquet_bytes(frame)
    manifest = manifest_for(blob, universe=620, instruments=2)
    archive = build_archive(
        manifest,
        None,
        frame,
        pd.DataFrame({"instrument_id": ["CZ00001"], "ticker": ["AAPL"], "name": ["Apple Inc."]}),
        datetime(2026, 9, 26, tzinfo=UTC),
    )
    slim, parts = split_archive(archive)
    assert slim["split"] is True
    assert "forecasts" not in slim["instruments"]["CZ00001"]  # type: ignore[index]
    assert slim["instruments"]["CZ00001"]["ticker"] == "AAPL"  # type: ignore[index]
    assert [f["horizon"] for f in parts["CZ00001"]["forecasts"]] == [5, 20, 60]  # type: ignore[index]
    assert len(json.dumps(slim)) < len(json.dumps(archive)) / 2
    # Az index sora a lenyomatból: ugyanaz, amit az archívumból olvastunk volna.
    entry = index_entry("2026-09-23", manifest, None)
    assert entry["instruments_with_forecast"] == archive["instruments_with_forecast"]
    assert entry["partial"] == archive["partial"]
    assert entry["sha256"] == archive["sha256"]
