"""Tests for validation module."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.data.synthetic_generator import generate_all
from src.analysis.regression import run_all_regressions
from src.analysis.validation import (
    Check, ValidationReport, validate_descriptive_stats,
    validate_correlations, validate_exchange_concentration,
    validate_monthly_decay, validate_post_pump,
    validate_rank_buckets, run_full_validation
)


@pytest.fixture(scope="module")
def datasets(): return generate_all()
@pytest.fixture(scope="module")
def dc(datasets): return datasets[0]
@pytest.fixture(scope="module")
def tg(datasets): return datasets[1]
@pytest.fixture(scope="module")
def reg_results(dc, tg): return run_all_regressions(dc, tg)


def test_check_pass():
    c = Check("test", 100.0, 105.0, 10.0, "Sec 1")
    assert c.passed

def test_check_fail():
    c = Check("test", 100.0, 200.0, 10.0, "Sec 1")
    assert not c.passed

def test_check_delta_computation():
    c = Check("test", 100.0, 110.0, 20.0, "Sec 1")
    assert abs(c.delta_pct - 10.0) < 0.01

def test_validation_report_pass_rate():
    r = ValidationReport()
    r.add(Check("a", 1.0, 1.0, 5.0, "X"))
    r.add(Check("b", 1.0, 2.0, 5.0, "X"))
    assert r.pass_rate == 50.0

def test_validation_report_to_dataframe():
    r = ValidationReport()
    r.add(Check("a", 1.0, 1.0, 5.0, "X"))
    df = r.to_dataframe()
    assert "check" in df.columns and "passed" in df.columns

def test_descriptive_stats_length(dc, tg):
    checks = validate_descriptive_stats(dc, tg)
    assert len(checks) == 10

def test_discord_n_check_passes(dc, tg):
    checks = validate_descriptive_stats(dc, tg)
    n_check = next(c for c in checks if c.name == "Discord N")
    assert n_check.passed

def test_correlations_length(dc, tg):
    checks = validate_correlations(dc, tg)
    assert len(checks) == 5

def test_exchange_concentration_checks(dc, tg):
    checks = validate_exchange_concentration(dc, tg)
    assert len(checks) == 4

def test_monthly_decay_checks_count(dc, tg):
    checks = validate_monthly_decay(dc, tg)
    assert len(checks) >= 12  # 6 months x 2 platforms + 2 decline checks

def test_post_pump_checks(dc, tg):
    checks = validate_post_pump(dc, tg)
    assert len(checks) == 4

def test_rank_bucket_checks(dc, tg):
    checks = validate_rank_buckets(dc, tg)
    assert len(checks) >= 6

def test_full_validation_runs(dc, tg, reg_results):
    report = run_full_validation(dc, tg, reg_results)
    assert len(report.checks) >= 35
    assert report.pass_rate >= 50.0  # should get majority right

def test_full_validation_summary_string(dc, tg):
    report = run_full_validation(dc, tg)
    summary = report.summary()
    assert "Validation Summary" in summary

def test_check_zero_paper_value():
    c = Check("test", 0.0, 0.5, 10.0, "X")
    assert c.delta_pct >= 0
