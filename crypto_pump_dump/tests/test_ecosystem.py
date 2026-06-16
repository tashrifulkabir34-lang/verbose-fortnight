"""Tests for ecosystem concentration analysis — Section 3.3."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from src.data.synthetic_generator import generate_all
from src.analysis.ecosystem import (
    channel_concentration, exchange_concentration,
    coin_repetition_analysis, monthly_profitability,
    profitability_decline_rate, rank_bucket_analysis, full_ecosystem_report
)


@pytest.fixture(scope="module")
def datasets(): return generate_all()
@pytest.fixture(scope="module")
def dc(datasets): return datasets[0]
@pytest.fixture(scope="module")
def tg(datasets): return datasets[1]


def test_channel_concentration_keys(tg):
    cc = channel_concentration(tg)
    assert "top_n_share_pct" in cc and "hhi" in cc

def test_telegram_top3_channel_share(tg):
    cc = channel_concentration(tg, top_n=3)
    assert 25 <= cc["top_n_share_pct"] <= 70

def test_hhi_in_valid_range(tg):
    cc = channel_concentration(tg)
    assert 0 < cc["hhi"] <= 1.0

def test_exchange_concentration_keys(dc):
    ec = exchange_concentration(dc)
    assert "binance_bittrex_combined_pct" in ec
    assert "pct_no_exchange" in ec

def test_discord_binance_bittrex_share_close_to_paper(dc):
    ec = exchange_concentration(dc)
    assert 70 <= ec["binance_bittrex_combined_pct"] <= 100

def test_no_exchange_share_reasonable(dc):
    ec = exchange_concentration(dc)
    assert 35 <= ec["pct_no_exchange"] <= 60

def test_coin_repetition_keys(tg):
    cr = coin_repetition_analysis(tg, threshold=18)
    assert "n_coins_above_threshold" in cr
    assert "pct_from_heavy_coins" in cr

def test_monthly_profitability_returns_dataframe(dc):
    mp = monthly_profitability(dc, "discord")
    assert len(mp) == 6
    assert "median_pct" in mp.columns

def test_monthly_profitability_all_months(tg):
    mp = monthly_profitability(tg, "telegram")
    assert set(mp.index) == {1,2,3,4,5,6}

def test_profitability_decline_rate_keys(dc):
    pd_info = profitability_decline_rate(dc)
    assert "decline_pct" in pd_info

def test_profitability_decline_direction(dc):
    pd_info = profitability_decline_rate(dc)
    assert pd_info["decline_pct"] < 0  # must be a decline

def test_rank_bucket_analysis_shape(dc):
    rb = rank_bucket_analysis(dc)
    assert len(rb) == 4

def test_rank_bucket_ordering(dc):
    rb = rank_bucket_analysis(dc)
    medians = rb["median_price_inc"].values
    assert medians[-1] > medians[0]  # obscure coins have higher spike

def test_full_ecosystem_report_structure(dc, tg):
    report = full_ecosystem_report(dc, tg)
    assert "discord" in report and "telegram" in report
    assert "channel_concentration" in report["discord"]
    assert "exchange_concentration" in report["telegram"]
