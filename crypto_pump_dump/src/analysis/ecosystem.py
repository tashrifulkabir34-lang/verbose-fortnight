"""
Ecosystem Concentration Analysis — Section 3.3 of Hamrick et al.
=================================================================
Replicates findings on:
  (I)   Channel concentration — top 3 channels = ~45% of Telegram pumps
  (II)  Exchange concentration — Binance+Bittrex = ~86-87%
  (III) Coin repetition — 23 coins pumped 18+ times on Telegram
  (IV)  Monthly profitability decay (Table 8)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ── I. Channel Concentration ──────────────────────────────────────────────────

def channel_concentration(df: pd.DataFrame, top_n: int = 3) -> Dict:
    """
    Compute HHI and top-N share of pump channels.
    Paper: top 3 Telegram channels ≈ 45% of pumps.
    """
    counts = df["channel"].value_counts()
    total = len(df)
    shares = counts / total

    top_n_share = shares.iloc[:top_n].sum()
    hhi = (shares ** 2).sum()

    return {
        "top_n": top_n,
        "top_n_share_pct": round(top_n_share * 100, 2),
        "hhi": round(float(hhi), 4),
        "n_unique_channels": int(counts.nunique()),
        "top_channels": counts.iloc[:top_n].to_dict(),
        "paper_top3_share_pct": 45.0,
    }


# ── II. Exchange Concentration ────────────────────────────────────────────────

def exchange_concentration(df: pd.DataFrame) -> Dict:
    """
    Compute exchange share.
    Paper: Binance + Bittrex = 86-87% of exchange-listed pumps.
    """
    listed = df[df["exchange"] != "None"].copy()
    counts = listed["exchange"].value_counts()
    total_listed = len(listed)

    exchange_shares = (counts / total_listed * 100).round(2).to_dict()

    binance_bittrex_share = exchange_shares.get("Binance", 0) + \
                            exchange_shares.get("Bittrex", 0) + \
                            exchange_shares.get("Binance+Bittrex", 0)

    pct_no_exchange = (df["exchange"] == "None").mean() * 100

    return {
        "exchange_shares_pct": exchange_shares,
        "binance_bittrex_combined_pct": round(binance_bittrex_share, 2),
        "pct_no_exchange": round(pct_no_exchange, 2),
        "total_pumps_with_exchange": int(total_listed),
        "paper_binance_bittrex_pct": 86.5,
    }


# ── III. Coin Repetition ──────────────────────────────────────────────────────

def coin_repetition_analysis(df: pd.DataFrame, threshold: int = 18) -> Dict:
    """
    Replicate finding: 23 coins pumped ≥18 times on Telegram.
    These accounted for >20% of all Telegram pumps.
    """
    coin_counts = df.groupby("coin_rank").size().sort_values(ascending=False)
    heavy_pumped = coin_counts[coin_counts >= threshold]

    total = len(df)
    heavy_share = heavy_pumped.sum() / total * 100

    return {
        "threshold": threshold,
        "n_coins_above_threshold": int(len(heavy_pumped)),
        "pumps_from_heavy_coins": int(heavy_pumped.sum()),
        "pct_from_heavy_coins": round(float(heavy_share), 2),
        "paper_n_coins": 23,
        "paper_pct_share": 20.0,
        "top_repeated_coins": heavy_pumped.head(10).to_dict(),
        "median_pumps_per_coin": round(float(coin_counts.median()), 2),
        "mean_pumps_per_coin": round(float(coin_counts.mean()), 2),
    }


# ── IV. Monthly Profitability Decay ───────────────────────────────────────────

def monthly_profitability(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Replicate Table 8: median pump success by month.
    """
    from src.data.synthetic_generator import MONTHLY_DECAY

    monthly = (
        df.groupby("month")["max_price_inc_pct"]
        .agg(["median", "mean", "count"])
        .rename(columns={"median": "median_pct", "mean": "mean_pct", "count": "n_pumps"})
        .round(2)
    )

    paper_key = platform.lower()
    paper_medians = MONTHLY_DECAY.get(paper_key, {})
    monthly["paper_median_pct"] = [paper_medians.get(m, np.nan) for m in monthly.index]
    monthly["delta_vs_paper"] = (monthly["median_pct"] - monthly["paper_median_pct"]).round(2)

    return monthly


def profitability_decline_rate(df: pd.DataFrame) -> Dict:
    """
    Measure % decline in median profitability from Jan to Jun.
    Paper: Discord -60%, Telegram -50%.
    """
    monthly_med = df.groupby("month")["max_price_inc_pct"].median()
    if len(monthly_med) < 2:
        return {}

    jan_val = monthly_med.get(1, np.nan)
    jun_val = monthly_med.get(6, np.nan)
    if pd.isna(jan_val) or pd.isna(jun_val) or jan_val == 0:
        return {}

    decline_pct = (jun_val - jan_val) / jan_val * 100

    return {
        "jan_median_pct": round(float(jan_val), 2),
        "jun_median_pct": round(float(jun_val), 2),
        "decline_pct": round(float(decline_pct), 2),
    }


# ── V. Coin Rank Buckets (Table 3) ────────────────────────────────────────────

def rank_bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate Table 3: median price increase by rank bucket.
    """
    bins = [0, 75, 200, 500, np.inf]
    labels = ["≤75", "76-200", "201-500", ">500"]

    df = df.copy()
    df["rank_bucket"] = pd.cut(df["coin_rank"], bins=bins, labels=labels)

    result = (
        df.groupby("rank_bucket", observed=False)
        .agg(
            n_pumps=("max_price_inc_pct", "count"),
            median_price_inc=("max_price_inc_pct", "median"),
            mean_price_inc=("max_price_inc_pct", "mean"),
            n_unique_coins=("coin_rank", "nunique"),
        )
        .round(2)
    )

    # Paper's Table 3 values
    paper_discord = {"≤75": 3.51, "76-200": 5.22, "201-500": 5.32, ">500": 23.23}
    paper_telegram = {"≤75": 4.81, "76-200": 6.46, "201-500": 8.10, ">500": 18.74}

    return result


# ── Full Ecosystem Report ─────────────────────────────────────────────────────

def full_ecosystem_report(dc_df: pd.DataFrame, tg_df: pd.DataFrame) -> Dict:
    """Run all ecosystem analyses and return consolidated report."""
    return {
        "discord": {
            "channel_concentration": channel_concentration(dc_df),
            "exchange_concentration": exchange_concentration(dc_df),
            "coin_repetition": coin_repetition_analysis(dc_df, threshold=10),
            "monthly_profitability": monthly_profitability(dc_df, "discord"),
            "profitability_decline": profitability_decline_rate(dc_df),
            "rank_buckets": rank_bucket_analysis(dc_df),
        },
        "telegram": {
            "channel_concentration": channel_concentration(tg_df),
            "exchange_concentration": exchange_concentration(tg_df),
            "coin_repetition": coin_repetition_analysis(tg_df, threshold=18),
            "monthly_profitability": monthly_profitability(tg_df, "telegram"),
            "profitability_decline": profitability_decline_rate(tg_df),
            "rank_buckets": rank_bucket_analysis(tg_df),
        },
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data.synthetic_generator import generate_all
    dc, tg = generate_all()
    report = full_ecosystem_report(dc, tg)

    print("Channel Concentration — Telegram")
    print(report["telegram"]["channel_concentration"])
    print("\nExchange Concentration — Discord")
    print(report["discord"]["exchange_concentration"])
    print("\nMonthly Profitability — Telegram")
    print(report["telegram"]["monthly_profitability"])
    print("\nRank Bucket Analysis — Discord")
    print(report["discord"]["rank_buckets"])
