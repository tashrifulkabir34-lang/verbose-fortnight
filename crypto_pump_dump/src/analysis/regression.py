"""
Regression Analysis — Replicating Table 6 of Hamrick et al.
============================================================
Log/log OLS with clustered standard errors at the coin level.

Dependent variable  : ln(Max % Price Increase)
Key independent vars: ln(Rank), ln(Exchanges), ln(PairCount), monthly dummies, exchange dummies
Clustering          : coin_rank decile (proxy for coin identity)
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, Any


def _build_log_cols(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    df = df.copy()
    df["ln_price_inc"]  = np.log(df["max_price_inc_pct"].clip(lower=0.001))
    df["ln_rank"]       = np.log(df["coin_rank"].clip(lower=1))
    df["ln_exchanges"]  = np.log(df["n_exchanges"].clip(lower=1))
    df["ln_pair_count"] = np.log(df["pair_count"].clip(lower=1))

    if platform == "Telegram":
        df["ln_views"] = np.log((df["views"] + 1).clip(lower=1))
    else:
        df["ln_server_members"] = np.log(df["server_members"].clip(lower=1))

    for m in range(2, 7):
        df[f"month_{m}"] = (df["month"] == m).astype(int)

    df["binance_only"]    = (df["exchange"] == "Binance").astype(int)
    df["bittrex_only"]    = (df["exchange"] == "Bittrex").astype(int)
    df["binance_bittrex"] = (df["exchange"] == "Binance+Bittrex").astype(int)
    df["other_exchange"]  = (df["exchange"] == "Other").astype(int)
    return df


def _build_formula(platform: str) -> str:
    base = (
        "ln_price_inc ~ ln_rank + ln_exchanges + ln_pair_count "
        "+ month_2 + month_3 + month_4 + month_5 + month_6 "
        "+ binance_only + bittrex_only + binance_bittrex + other_exchange"
    )
    return base + (" + ln_views" if platform == "Telegram" else " + ln_server_members")


def run_ols_clustered(df: pd.DataFrame, platform: str) -> Dict[str, Any]:
    df_log = _build_log_cols(df, platform)
    formula = _build_formula(platform)
    model  = smf.ols(formula, data=df_log)
    df_log["cluster_group"] = pd.qcut(df_log["coin_rank"], q=50, labels=False, duplicates="drop")
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df_log["cluster_group"].values})

    # Expected signs from Table 6
    expected_signs = {
        "ln_rank": "+", "ln_exchanges": "-", "ln_pair_count": "+",
        "month_2": "-", "month_3": "-", "month_4": "-", "month_5": "-", "month_6": "-",
        "binance_only": "-", "bittrex_only": "-", "binance_bittrex": "-",
    }

    sign_checks = {}
    for var, expected in expected_signs.items():
        if var in result.params:
            actual_sign = "+" if result.params[var] > 0 else "-"
            sign_checks[var] = {
                "expected": expected,
                "actual": actual_sign,
                "coef": round(float(result.params[var]), 4),
                "pval": round(float(result.pvalues[var]), 4),
                "match": actual_sign == expected,
            }

    return {
        "platform": platform,
        "result": result,
        "adj_r2": round(float(result.rsquared_adj), 4),
        "n_obs": int(result.nobs),
        "sign_checks": sign_checks,
        "formula": formula,
    }


def compare_adj_r2(discord_res: Dict, telegram_res: Dict) -> Dict[str, float]:
    return {
        "discord_adj_r2_paper": 0.30,
        "discord_adj_r2_replicated": discord_res["adj_r2"],
        "discord_r2_delta": round(discord_res["adj_r2"] - 0.30, 4),
        "telegram_adj_r2_paper": 0.32,
        "telegram_adj_r2_replicated": telegram_res["adj_r2"],
        "telegram_r2_delta": round(telegram_res["adj_r2"] - 0.32, 4),
    }


def extract_coef_table(result_dict: Dict) -> pd.DataFrame:
    res = result_dict["result"]
    df = pd.DataFrame({
        "coefficient": res.params,
        "std_error": res.bse,
        "t_stat": res.tvalues,
        "p_value": res.pvalues,
    }).round(4)
    df["significance"] = df["p_value"].apply(
        lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
    )
    df.index.name = "variable"
    return df


def run_post_pump_analysis(df: pd.DataFrame, platform: str) -> Dict[str, float]:
    post = df["post_pump_change_pct"]
    pct_negative = (post < 0).mean() * 100
    paper_median = -41.0 if platform == "Discord" else -38.0
    return {
        "median_post_pump_pct": round(float(post.median()), 2),
        "mean_post_pump_pct":   round(float(post.mean()), 2),
        "pct_below_pre_pump":   round(float(pct_negative), 1),
        "paper_median":         paper_median,
        "paper_pct_negative":   60.0,
    }


def run_all_regressions(dc_df: pd.DataFrame, tg_df: pd.DataFrame) -> Dict:
    dc_res = run_ols_clustered(dc_df, "Discord")
    tg_res = run_ols_clustered(tg_df, "Telegram")
    return {
        "discord": dc_res,
        "telegram": tg_res,
        "r2_comparison": compare_adj_r2(dc_res, tg_res),
        "post_pump": {
            "discord":  run_post_pump_analysis(dc_df, "Discord"),
            "telegram": run_post_pump_analysis(tg_df, "Telegram"),
        },
    }
