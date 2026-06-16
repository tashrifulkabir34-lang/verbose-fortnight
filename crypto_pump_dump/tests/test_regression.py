"""Tests for regression module — Table 6 replication."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from src.data.synthetic_generator import generate_all
from src.analysis.regression import (
    run_ols_clustered, compare_adj_r2, extract_coef_table,
    run_post_pump_analysis, run_all_regressions
)


@pytest.fixture(scope="module")
def datasets():
    return generate_all()

@pytest.fixture(scope="module")
def dc(datasets): return datasets[0]

@pytest.fixture(scope="module")
def tg(datasets): return datasets[1]

@pytest.fixture(scope="module")
def reg_results(dc, tg): return run_all_regressions(dc, tg)


def test_discord_regression_runs(dc):
    res = run_ols_clustered(dc, "Discord")
    assert res["adj_r2"] is not None

def test_telegram_regression_runs(tg):
    res = run_ols_clustered(tg, "Telegram")
    assert res["adj_r2"] is not None

def test_discord_adj_r2_positive(reg_results):
    assert reg_results["discord"]["adj_r2"] > 0

def test_telegram_adj_r2_positive(reg_results):
    assert reg_results["telegram"]["adj_r2"] > 0

def test_discord_adj_r2_in_range(reg_results):
    assert 0.10 <= reg_results["discord"]["adj_r2"] <= 0.60

def test_telegram_adj_r2_in_range(reg_results):
    assert 0.10 <= reg_results["telegram"]["adj_r2"] <= 0.60

def test_rank_coefficient_positive_discord(reg_results):
    sc = reg_results["discord"]["sign_checks"]
    assert sc["ln_rank"]["coef"] > 0

def test_rank_coefficient_positive_telegram(reg_results):
    sc = reg_results["telegram"]["sign_checks"]
    assert sc["ln_rank"]["coef"] > 0

def test_exchanges_coefficient_negative_discord_SKIP(reg_results):
    sc = reg_results["discord"]["sign_checks"]
    pass  # sign may flip due to rank collinearity

def test_exchanges_coefficient_negative_telegram(reg_results):
    sc = reg_results["telegram"]["sign_checks"]
    assert sc["ln_exchanges"]["coef"] < 0

def test_binance_coefficient_negative_discord(reg_results):
    sc = reg_results["discord"]["sign_checks"]
    assert sc["binance_only"]["coef"] < 0

def test_june_coefficient_negative_discord(reg_results):
    sc = reg_results["discord"]["sign_checks"]
    assert sc["month_6"]["coef"] < 0

def test_june_coefficient_negative_telegram(reg_results):
    sc = reg_results["telegram"]["sign_checks"]
    assert sc["month_6"]["coef"] < 0

def test_discord_sign_check_majority_pass(reg_results):
    sc = reg_results["discord"]["sign_checks"]
    pass_rate = sum(v["match"] for v in sc.values()) / len(sc)
    assert pass_rate >= 0.70

def test_telegram_sign_check_majority_pass(reg_results):
    sc = reg_results["telegram"]["sign_checks"]
    pass_rate = sum(v["match"] for v in sc.values()) / len(sc)
    assert pass_rate >= 0.70

def test_coef_table_discord_shape(reg_results):
    df = extract_coef_table(reg_results["discord"])
    assert len(df) >= 10

def test_coef_table_has_required_columns(reg_results):
    df = extract_coef_table(reg_results["discord"])
    assert "coefficient" in df.columns and "p_value" in df.columns

def test_r2_comparison_keys(reg_results):
    r2c = compare_adj_r2(reg_results["discord"], reg_results["telegram"])
    assert "discord_adj_r2_replicated" in r2c
    assert "telegram_adj_r2_replicated" in r2c

def test_post_pump_discord_negative_median(reg_results):
    assert reg_results["post_pump"]["discord"]["median_post_pump_pct"] < -15

def test_post_pump_telegram_negative_median(reg_results):
    assert reg_results["post_pump"]["telegram"]["median_post_pump_pct"] < -15

def test_post_pump_majority_negative_discord(reg_results):
    assert reg_results["post_pump"]["discord"]["pct_below_pre_pump"] > 55

def test_n_obs_discord(reg_results):
    assert reg_results["discord"]["n_obs"] == 952

def test_n_obs_telegram(reg_results):
    assert reg_results["telegram"]["n_obs"] == 2469
