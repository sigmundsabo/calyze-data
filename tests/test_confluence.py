"""Konfluencia-motor (`docs/konfluencia.md`).

Szintetikus adaton, ahol tudjuk a választ: egy kombináció, ami mindkét
időszakban tényleg működik, „confirmed”; ami csak a felfedezésben, „found,
not confirmed”; aminek kevés az eseménye, „too early”. És a kontextusok nem
látnak a jövőbe.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from pipeline.confluence.contexts import (
    BIT,
    CONTEXTS,
    combo_id,
    excluded,
    instrument_masks,
    is_trigger,
    subsets,
)
from pipeline.confluence.run import CONFIRMATION_START, DISCOVERY_END, measure, summary_of
from pipeline.patterns.common import pivots
from pipeline.patterns.levels import walk_levels


def random_walk(seed: int, start: str = "2010-01-01", n: int = 3600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    volume = rng.lognormal(13, 0.4, n)
    days = pd.bdate_range(start, periods=n).date
    return pd.DataFrame(
        {"date": days, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------- a család


def test_mely_szabaly_kivalto() -> None:
    assert is_trigger("hammer")
    assert not is_trigger("hammer|at_level")
    assert is_trigger("hs_top|retested")
    assert is_trigger("fib_up_golden|wick")
    assert not is_trigger("fib_up_golden|body")
    assert is_trigger("structure_break_up")
    assert not is_trigger("structure_break_up|bos")


def test_a_kivalto_nem_parosul_onmagaval() -> None:
    assert "level" in excluded("hammer")
    assert "level" in excluded("sr_support_touch")
    assert "htf" in excluded("topdown_up")
    assert "trend" in excluded("close_above_ema200")
    assert "structure" in excluded("structure_break_down")
    assert excluded("rsi_oversold") == frozenset()


def test_legfeljebb_negy_feltetel_es_41_reszhalmaz() -> None:
    full = subsets("rsi_oversold")
    assert len(full) == 6 + 15 + 20  # C(6,1)+C(6,2)+C(6,3)
    assert max(len(s) for s in full) == 3  # + a kiváltó = 4 feltétel
    assert len(subsets("hammer")) == 5 + 10 + 10  # a „level” nélkül
    assert combo_id("hammer", ("trend", "calm")) == "hammer+trend+calm"


# ---------------------------------------------------------------- jövőbe-nézés


def test_a_kontextus_a_multon_elvagva_ugyanaz() -> None:
    frame = random_walk(5, n=1500)
    regime = {d: ("stressed" if i % 7 == 0 else "calm") for i, d in enumerate(frame["date"])}

    def masks(f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return instrument_masks(f, walk_levels(f), pivots(f), regime)

    full_long, full_short = masks(frame)
    for cut in (600, 1100):
        part_long, part_short = masks(frame.iloc[:cut].reset_index(drop=True))
        np.testing.assert_array_equal(part_long, full_long[:cut])
        np.testing.assert_array_equal(part_short, full_short[:cut])
    # minden kontextus legalább egyszer igaz, különben a teszt semmit nem bizonyítana
    seen = np.bitwise_or.reduce(np.concatenate([full_long, full_short]))
    assert all(seen & BIT[c] for c in CONTEXTS)


# ---------------------------------------------------------------- a mérés


def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Árak 2010–2023, és három kiváltó, ahol tudjuk a választ az 5 napos horizonton."""
    frames = []
    for i in range(12):
        f = random_walk(100 + i)
        f["instrument_id"] = f"CZ{i:05d}"
        frames.append(f)
    prices = pd.concat(frames, ignore_index=True)

    rows = []
    rng = np.random.default_rng(0)
    trend = BIT["trend"]
    for instrument, g in prices.groupby("instrument_id"):
        c = g["close"].to_numpy()
        days = g["date"].to_numpy()
        future_up = np.r_[c[5:] > c[:-5], np.zeros(5, dtype=bool)]
        for t in rng.choice(len(g) - 10, size=400, replace=False):
            d = days[t]
            discovery = d <= DISCOVERY_END
            # „works”: mindkét időszakban a jövő irányába szól, ha áll a trend
            if future_up[t]:
                rows.append((instrument, d, "works", "long", trend))
            # „fades”: csak a felfedezésben szól jó irányba, utána rossz irányba
            if future_up[t] == discovery:
                rows.append((instrument, d, "fades", "long", trend))
        for t in rng.choice(len(g) - 10, size=1, replace=False):
            rows.append((instrument, days[t], "rare", "long", trend))
    events = pd.DataFrame(rows, columns=["instrument_id", "date", "trigger", "direction", "mask"])
    return events, prices


def test_mukodo_eltuno_es_ritka_kombinacio() -> None:
    events, prices = synthetic()
    table = measure(events, prices, workers=1)
    five = table[(table["horizon"] == 5) & (table["contexts"] == "trend")].set_index("trigger")

    assert five.loc["works", "status"] == "confirmed"
    assert five.loc["fades", "status"] == "found_not_confirmed"
    assert five.loc["rare", "status"] == "too_early"
    # a kontextus nélküli trend-bit kombináció más kontextust nem kér: a „trend+calm” üres
    assert (
        table[(table["trigger"] == "works") & (table["contexts"] == "trend+calm")]["status"]
        .eq("too_early")
        .all()
    )


def test_a_felfedezes_kimenetele_nem_nyulik_at_a_hataron() -> None:
    events, prices = synthetic()
    late = events[events["date"] > date(2018, 12, 20)]
    early = events[events["date"] <= date(2018, 12, 20)]
    # ha csak a határ utáni eseményeket adjuk, a felfedezésben nincs mit mérni
    table = measure(late[late["trigger"] == "works"], prices, workers=1)
    assert (table["status"] == "too_early").all()
    assert not early.empty
    assert CONFIRMATION_START > DISCOVERY_END


def test_az_osszesites_csak_darabszam() -> None:
    events, prices = synthetic()
    summary = summary_of(measure(events, prices, workers=1), events)
    assert set(summary) == {"events", "triggers", "defined", "tested", "found", "confirmed"}
    assert all(isinstance(v, int) for v in summary.values())
    assert summary["confirmed"] >= 1


# ---------------------------------------------------------------- élő panel


def test_a_panel_csak_a_ma_allo_kombinaciokat_hozza() -> None:
    from pipeline.confluence.live import panel

    results = pd.DataFrame(
        [
            {
                "combo": "hammer+trend",
                "horizon": 20,
                "status": "not_found",
                "disc_n": 400,
                "disc_hit": 0.55,
                "disc_baseline": 0.55,
                "disc_delta": 0.0,
                "disc_n_eff": 200.0,
                "disc_p": 0.9,
                "disc_verdict": "same",
            },
            {
                "combo": "hammer+trend+calm",
                "horizon": 20,
                "status": "found_not_confirmed",
                "disc_n": 300,
                "disc_hit": 0.6,
                "disc_baseline": 0.55,
                "disc_delta": 0.05,
                "disc_n_eff": 150.0,
                "disc_p": 0.01,
                "disc_verdict": "better_significant",
                "conf_n": 120,
                "conf_hit": 0.54,
                "conf_baseline": 0.56,
                "conf_delta": -0.02,
                "conf_n_eff": 60.0,
                "conf_p": 0.8,
                "conf_verdict": "worse",
            },
            # a „volume” ma nem áll: nem kerülhet a panelre
            {
                "combo": "hammer+volume",
                "horizon": 20,
                "status": "not_found",
                "disc_n": 90,
                "disc_hit": 0.5,
                "disc_baseline": 0.55,
                "disc_delta": -0.05,
                "disc_n_eff": 50.0,
                "disc_p": 0.5,
                "disc_verdict": "worse",
            },
            {"combo": "hammer+calm", "horizon": 20, "status": "too_early", "disc_n": 12},
        ]
    )
    out = panel([("hammer", "long", ("trend", "calm"))], results, date(2026, 9, 25))
    assert out["measured"] is True
    assert out["tested"] == 3
    combos = {c["combo"]: c for c in out["active"][0]["combos"]}
    assert set(combos) == {"hammer+trend", "hammer+trend+calm", "hammer+calm"}
    assert combos["hammer+trend+calm"]["confirmation"]["verdict"] == "worse"
    assert combos["hammer+trend"]["confirmation"] is None
    assert combos["hammer+calm"]["discovery"] == {
        "n": 12,
        "n_eff": None,
        "hit": None,
        "baseline": None,
        "delta": None,
        "p": None,
        "verdict": None,
    }


def test_meres_nelkul_kimondja() -> None:
    from pipeline.confluence.live import panel

    out = panel([("hammer", "long", ("trend",))], None, date(2026, 9, 25))
    assert out == {"session": "2026-09-25", "measured": False, "tested": 0, "active": []}


def test_a_mai_kivaltok_ugyanazok_mint_a_teljes_mulon() -> None:
    from pipeline.confluence.live import standing
    from pipeline.confluence.run import instrument_triggers

    frame = random_walk(9, n=1200)
    regime = {d: "calm" for d in frame["date"]}
    empty = pd.DataFrame(columns=["date", "rule", "direction"])
    events = instrument_triggers("CZ00009", frame, empty, regime)
    busiest = events["date"].value_counts().index[0]
    got = standing("CZ00009", frame[frame["date"] <= busiest].reset_index(drop=True), empty, regime, busiest)
    assert {t for t, _, _ in got} == set(events.loc[events["date"] == busiest, "trigger"])


def test_a_csomag_szetbontva_a_tabla_csak_a_megerosito_sorokat_kapja():
    from pipeline.confluence.run import BOARD_FILE, split_payload

    payload = {
        "generated_at": "x",
        "horizons": [5, 20],
        "confirmation": {"from": "2019-01-01", "to": "2026-01-01"},
        "rows": [
            {
                "combo": "a",
                "trigger": "t",
                "contexts": [],
                "horizon": 5,
                "status": "confirmed",
                "discovery": {"n": 1},
                "confirmation": {"n": 40},
            },
            {
                "combo": "b",
                "trigger": "t",
                "contexts": [],
                "horizon": 20,
                "status": "not_found",
                "discovery": {"n": 1},
            },
        ],
    }
    parts = split_payload(payload)
    assert [r["combo"] for r in parts[BOARD_FILE]["rows"]] == ["a"]
    assert "discovery" not in parts[BOARD_FILE]["rows"][0]
    assert [r["combo"] for r in parts["confluence/h5.json"]["rows"]] == ["a"]
    assert [r["combo"] for r in parts["confluence/h20.json"]["rows"]] == ["b"]
