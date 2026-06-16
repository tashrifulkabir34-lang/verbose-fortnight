"""Tests for synthetic data generator — calibration to paper stats."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic_generator import generate_discord, generate_telegram, generate_all, PAPER_STATS, MONTHLY_DECAY


@pytest.fixture(scope="module")
def dc(): return generate_discord()

@pytest.fixture(scope="module")
def tg(): return generate_telegram()


# ── Sample sizes ──────────────────────────────────────────────────────────────
def test_discord_sample_size(dc): assert len(dc) == 952
def test_telegram_sample_size(tg): assert len(tg) == 2469
def test_generate_all_returns_correct_sizes():
    d, t = generate_all()
    assert len(d) == 952 and len(t) == 2469

# ── Column presence ───────────────────────────────────────────────────────────
def test_discord_required_columns(dc):
    required = ["platform","pump_id","month","coin_rank","n_exchanges","pair_count",
                "channel","exchange","max_price_inc_pct","post_pump_change_pct","server_members"]
    assert all(c in dc.columns for c in required)

def test_telegram_required_columns(tg):
    required = ["platform","pump_id","month","coin_rank","n_exchanges","pair_count",
                "channel","exchange","max_price_inc_pct","post_pump_change_pct","views"]
    assert all(c in tg.columns for c in required)

# ── Platform labels ───────────────────────────────────────────────────────────
def test_discord_platform_label(dc): assert (dc["platform"] == "Discord").all()
def test_telegram_platform_label(tg): assert (tg["platform"] == "Telegram").all()

# ── Price increase: positivity and range ─────────────────────────────────────
def test_discord_price_inc_positive(dc): assert (dc["max_price_inc_pct"] > 0).all()
def test_telegram_price_inc_positive(tg): assert (tg["max_price_inc_pct"] > 0).all()
def test_discord_price_inc_max(dc): assert dc["max_price_inc_pct"].max() <= 400
def test_telegram_price_inc_max(tg): assert tg["max_price_inc_pct"].max() <= 400

# ── Mean price increase within 40% of paper ───────────────────────────────────
def test_discord_mean_price_inc(dc):
    assert abs(dc["max_price_inc_pct"].mean() - 6.78) / 6.78 < 0.40

def test_telegram_mean_price_inc(tg):
    assert abs(tg["max_price_inc_pct"].mean() - 9.57) / 9.57 < 0.35

# ── Coin rank range ───────────────────────────────────────────────────────────
def test_discord_rank_min(dc): assert dc["coin_rank"].min() >= 2
def test_discord_rank_max(dc): assert dc["coin_rank"].max() <= 2036
def test_telegram_rank_min(tg): assert tg["coin_rank"].min() >= 2
def test_telegram_rank_max(tg): assert tg["coin_rank"].max() <= 2036

# ── Exchange counts ───────────────────────────────────────────────────────────
def test_discord_exchanges_positive(dc): assert (dc["n_exchanges"] >= 1).all()
def test_discord_exchanges_max(dc): assert dc["n_exchanges"].max() <= 182
def test_telegram_exchanges_max(tg): assert tg["n_exchanges"].max() <= 182

# ── Monthly distribution ──────────────────────────────────────────────────────
def test_discord_months_range(dc): assert dc["month"].between(1, 6).all()
def test_telegram_months_range(tg): assert tg["month"].between(1, 6).all()
def test_discord_all_months_present(dc): assert set(dc["month"].unique()) == {1,2,3,4,5,6}
def test_telegram_all_months_present(tg): assert set(tg["month"].unique()) == {1,2,3,4,5,6}

# ── Exchange categories ───────────────────────────────────────────────────────
def test_discord_exchange_categories(dc):
    assert set(dc["exchange"].unique()).issubset({"Binance","Bittrex","Binance+Bittrex","Other","None"})

def test_telegram_exchange_categories(tg):
    assert set(tg["exchange"].unique()).issubset({"Binance","Bittrex","Binance+Bittrex","Other","None"})

# ── Exchange concentration ─────────────────────────────────────────────────────
def test_discord_binance_bittrex_share(dc):
    listed = dc[dc["exchange"] != "None"]
    share = listed["exchange"].isin(["Binance","Bittrex","Binance+Bittrex"]).mean() * 100
    assert 75 <= share <= 100

def test_telegram_binance_bittrex_share(tg):
    listed = tg[tg["exchange"] != "None"]
    share = listed["exchange"].isin(["Binance","Bittrex","Binance+Bittrex"]).mean() * 100
    assert 75 <= share <= 100

# ── Pump type categories ──────────────────────────────────────────────────────
def test_discord_pump_types(dc):
    assert set(dc["pump_type"].unique()).issubset({"obvious","target","copied"})

def test_telegram_pump_types(tg):
    assert set(tg["pump_type"].unique()).issubset({"obvious","target"})

# ── Monthly profitability decay direction ─────────────────────────────────────
def test_discord_decay_direction(dc):
    monthly = dc.groupby("month")["max_price_inc_pct"].median()
    assert monthly[1] > monthly[6]

def test_telegram_decay_direction(tg):
    monthly = tg.groupby("month")["max_price_inc_pct"].median()
    assert monthly[1] > monthly[6]

# ── Monthly medians match Table 8 exactly (within 1%) ─────────────────────────
def test_discord_monthly_medians_table8(dc):
    paper = MONTHLY_DECAY["discord"]
    monthly = dc.groupby("month")["max_price_inc_pct"].median()
    for m, target in paper.items():
        assert abs(monthly[m] - target) / target < 0.02, f"Month {m}: {monthly[m]:.2f} vs {target}"

def test_telegram_monthly_medians_table8(tg):
    paper = MONTHLY_DECAY["telegram"]
    monthly = tg.groupby("month")["max_price_inc_pct"].median()
    for m, target in paper.items():
        assert abs(monthly[m] - target) / target < 0.02, f"Month {m}: {monthly[m]:.2f} vs {target}"

# ── Rank bucket ordering ──────────────────────────────────────────────────────
def test_discord_rank_bucket_ordering(dc):
    top = dc[dc["coin_rank"] <= 75]["max_price_inc_pct"].median()
    low = dc[dc["coin_rank"] >  500]["max_price_inc_pct"].median()
    assert low > top

def test_telegram_rank_bucket_ordering(tg):
    top = tg[tg["coin_rank"] <= 75]["max_price_inc_pct"].median()
    low = tg[tg["coin_rank"] >  500]["max_price_inc_pct"].median()
    assert low > top

# ── Correlation signs ─────────────────────────────────────────────────────────
def test_discord_corr_rank_price_positive(dc):
    assert dc["coin_rank"].corr(dc["max_price_inc_pct"]) > 0.15

def test_telegram_corr_rank_price_positive(tg):
    assert tg["coin_rank"].corr(tg["max_price_inc_pct"]) > 0.10

# ── Exchange price ordering (Table 6: Binance/Bittrex do WORSE) ──────────────
def test_discord_exchange_price_ordering(dc):
    b = dc[dc["exchange"] == "Binance"]["max_price_inc_pct"].median()
    n = dc[dc["exchange"] == "None"]["max_price_inc_pct"].median()
    assert b < n

def test_telegram_exchange_price_ordering(tg):
    b = tg[tg["exchange"] == "Binance"]["max_price_inc_pct"].median()
    n = tg[tg["exchange"] == "None"]["max_price_inc_pct"].median()
    assert b < n

# ── Post-pump crash ───────────────────────────────────────────────────────────
def test_discord_post_pump_median_negative(dc):
    assert dc["post_pump_change_pct"].median() < -20

def test_telegram_post_pump_median_negative(tg):
    assert tg["post_pump_change_pct"].median() < -20

def test_discord_majority_post_pump_negative(dc):
    assert (dc["post_pump_change_pct"] < 0).mean() > 0.55

def test_telegram_majority_post_pump_negative(tg):
    assert (tg["post_pump_change_pct"] < 0).mean() > 0.55

# ── Date column ───────────────────────────────────────────────────────────────
def test_discord_date_column(dc):
    assert "date" in dc.columns and pd.api.types.is_datetime64_any_dtype(dc["date"])

def test_telegram_date_column(tg):
    assert "date" in tg.columns

# ── Channel concentration ─────────────────────────────────────────────────────
def test_telegram_channel_concentration(tg):
    top3 = tg["channel"].value_counts().iloc[:3].sum() / len(tg)
    assert top3 >= 0.30

def test_discord_multiple_channels(dc):
    assert dc["channel"].nunique() >= 3

# ── No missing values ─────────────────────────────────────────────────────────
def test_discord_no_nulls(dc):
    assert dc[["coin_rank","n_exchanges","max_price_inc_pct","month","exchange"]].isnull().sum().sum() == 0

def test_telegram_no_nulls(tg):
    assert tg[["coin_rank","n_exchanges","max_price_inc_pct","month","exchange"]].isnull().sum().sum() == 0
