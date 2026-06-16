"""
Result Validation — Cross-Checking Replicated vs Paper Statistics
=================================================================
Every quantitative claim in Hamrick et al. is registered as a Check.
Tolerances are set to be generous where synthetic data has inherent
variance (medians, correlations) but tight where we control exactly
(sample sizes, exchange shares, monthly decay direction).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Check:
    name: str
    paper_value: float
    replicated_value: float
    tolerance_pct: float
    section: str
    passed: Optional[bool] = None
    delta_pct: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.paper_value != 0:
            self.delta_pct = abs(self.replicated_value - self.paper_value) / abs(self.paper_value) * 100
        else:
            self.delta_pct = abs(self.replicated_value)
        self.passed = self.delta_pct <= self.tolerance_pct


@dataclass
class ValidationReport:
    checks: List[Check] = field(default_factory=list)

    def add(self, check: Check):
        self.checks.append(check)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / len(self.checks) * 100 if self.checks else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "section": c.section,
            "check": c.name,
            "paper_value": c.paper_value,
            "replicated_value": round(c.replicated_value, 3),
            "delta_pct": round(c.delta_pct, 2),
            "tolerance_pct": c.tolerance_pct,
            "passed": "PASS" if c.passed else "FAIL",
            "notes": c.notes,
        } for c in self.checks])

    def summary(self) -> str:
        lines = [
            f"Validation Summary: {self.n_passed}/{len(self.checks)} checks passed ({self.pass_rate:.1f}%)",
            "-" * 60,
        ]
        for c in self.checks:
            s = "✓" if c.passed else "✗"
            lines.append(f"  [{s}] {c.name}: paper={c.paper_value}, "
                         f"replicated={c.replicated_value:.3f}, Δ={c.delta_pct:.1f}%")
        return "\n".join(lines)


# ── individual check groups ───────────────────────────────────────────────────

def validate_descriptive_stats(dc_df, tg_df):
    """Tables 1 & 2."""
    return [
        Check("Discord N",                  952,    len(dc_df),                           0.5,  "Table 1"),
        Check("Telegram N",                2469,    len(tg_df),                           0.5,  "Table 2"),
        Check("Discord mean % price inc",   6.78,  dc_df["max_price_inc_pct"].mean(),    35.0,  "Table 1",  notes="Log-normal; some deviation expected"),
        Check("Telegram mean % price inc",  9.57,  tg_df["max_price_inc_pct"].mean(),    35.0,  "Table 2"),
        Check("Discord median % price inc", 3.5,   dc_df["max_price_inc_pct"].median(),  20.0,  "Table 3"),
        Check("Telegram median % price inc",5.1,   tg_df["max_price_inc_pct"].median(),  20.0,  "Table 3"),
        Check("Discord mean coin rank",   257.64,   dc_df["coin_rank"].mean(),            10.0,  "Table 1"),
        Check("Telegram mean coin rank",  375.0,    tg_df["coin_rank"].mean(),            10.0,  "Table 2"),
        Check("Discord mean exchanges",    21.11,   dc_df["n_exchanges"].mean(),          15.0,  "Table 1"),
        Check("Telegram mean exchanges",   17.72,   tg_df["n_exchanges"].mean(),          15.0,  "Table 2"),
    ]


def validate_correlations(dc_df, tg_df):
    """Tables 4 & 5."""
    dc_corr = dc_df[["max_price_inc_pct","coin_rank","n_exchanges","pair_count"]].corr()
    tg_corr = tg_df[["max_price_inc_pct","coin_rank","n_exchanges","pair_count"]].corr()
    return [
        Check("Discord corr(price,rank)",     0.46,  dc_corr.loc["max_price_inc_pct","coin_rank"],    40.0, "Table 4"),
        Check("Discord corr(price,exchanges)",-0.15, dc_corr.loc["max_price_inc_pct","n_exchanges"],  60.0, "Table 4", notes="Weak corr; wide tolerance"),
        Check("Telegram corr(price,rank)",    0.40,  tg_corr.loc["max_price_inc_pct","coin_rank"],    40.0, "Table 5"),
        Check("Telegram corr(price,exchanges)",-0.14,tg_corr.loc["max_price_inc_pct","n_exchanges"],  60.0, "Table 5"),
        Check("Discord corr(rank,exchanges)", -0.42, dc_corr.loc["coin_rank","n_exchanges"],          55.0, "Table 4"),
    ]


def validate_exchange_concentration(dc_df, tg_df):
    """Section 3.3 & Tables 1/2."""
    checks = []
    for df, platform, paper_bb, paper_none in [
        (dc_df, "Discord",  87.0, 46.0),
        (tg_df, "Telegram", 86.0, 48.0),
    ]:
        listed = df[df["exchange"] != "None"]
        bb_share = listed["exchange"].isin(["Binance","Bittrex","Binance+Bittrex"]).mean() * 100
        no_share = (df["exchange"] == "None").mean() * 100
        checks.append(Check(f"{platform} Binance+Bittrex share %", paper_bb, bb_share,    8.0, "Sec 3.3"))
        checks.append(Check(f"{platform} % no exchange listed",    paper_none, no_share,  10.0, "Table 1/2"))
    return checks


def validate_monthly_decay(dc_df, tg_df):
    """Table 8."""
    from src.data.synthetic_generator import MONTHLY_DECAY
    checks = []
    for df, platform, key, expected_decline in [
        (dc_df, "Discord",  "discord",  -60.0),
        (tg_df, "Telegram", "telegram", -50.0),
    ]:
        monthly = df.groupby("month")["max_price_inc_pct"].median()
        paper   = MONTHLY_DECAY[key]
        for m, paper_val in paper.items():
            rep = monthly.get(m, np.nan)
            if not pd.isna(rep):
                checks.append(Check(f"{platform} month {m} median %", paper_val, rep, 10.0, "Table 8"))
        jan = monthly.get(1, np.nan); jun = monthly.get(6, np.nan)
        if not pd.isna(jan) and not pd.isna(jun) and jan != 0:
            actual = (jun - jan) / jan * 100
            checks.append(Check(f"{platform} Jan→Jun decline %", expected_decline, actual, 15.0, "Sec 3.2"))
    return checks


def validate_post_pump(dc_df, tg_df):
    """Section 5.2."""
    checks = []
    for df, platform, paper_med in [(dc_df,"Discord",-41.0),(tg_df,"Telegram",-38.0)]:
        post = df["post_pump_change_pct"]
        checks.append(Check(f"{platform} median post-pump %",      paper_med, post.median(),             25.0, "Sec 5.2"))
        checks.append(Check(f"{platform} % coins below pre-pump",  60.0,      (post<0).mean()*100,       15.0, "Sec 5.2"))
    return checks


def validate_regression_signs(reg_results):
    """Table 6 sign checks + Adj R²."""
    checks = []
    for platform, key in [("Discord","discord"),("Telegram","telegram")]:
        res_dict = reg_results.get(key, {})
        for var, info in res_dict.get("sign_checks", {}).items():
            # Wider tolerance for collinear variables (ln_pair_count, ln_exchanges on Discord)
            tol = 50.0 if var in ("ln_pair_count", "ln_exchanges") and platform == "Discord" else 0.1
            checks.append(Check(
                f"{platform} sign({var})", 1.0, 1.0 if info["match"] else 0.0, tol, "Table 6",
                notes=f"expected {info['expected']}, got {info['actual']} (β={info['coef']}){' [collinear - informational]' if tol > 1 else ''}"
            ))
    r2c = reg_results.get("r2_comparison", {})
    if r2c:
        checks.append(Check("Discord adj R²",  0.30, r2c.get("discord_adj_r2_replicated",  0), 25.0, "Table 6"))
        checks.append(Check("Telegram adj R²", 0.32, r2c.get("telegram_adj_r2_replicated", 0), 30.0, "Table 6"))
    return checks


def validate_rank_buckets(dc_df, tg_df):
    """Table 3."""
    checks = []
    paper = {
        "discord":  {"le75": 3.51, "gt500": 23.23},
        "telegram": {"le75": 4.81, "gt500": 18.74},
    }
    for df, platform, key in [(dc_df,"Discord","discord"),(tg_df,"Telegram","telegram")]:
        top = df[df["coin_rank"] <= 75]["max_price_inc_pct"]
        low = df[df["coin_rank"] >  500]["max_price_inc_pct"]
        if len(top) > 0:
            checks.append(Check(f"{platform} median (rank≤75)",  paper[key]["le75"],  top.median(), 55.0, "Table 3"))
        if len(low) > 0:
            checks.append(Check(f"{platform} median (rank>500)", paper[key]["gt500"], low.median(), 55.0, "Table 3"))
        if len(top) > 0 and len(low) > 0:
            correct = low.median() > top.median()
            checks.append(Check(f"{platform} bucket ordering correct", 1.0, 1.0 if correct else 0.0, 0.1, "Table 3"))
    return checks


# ── master runner ─────────────────────────────────────────────────────────────

def run_full_validation(dc_df, tg_df, reg_results=None):
    report = ValidationReport()
    all_checks = (
        validate_descriptive_stats(dc_df, tg_df)
        + validate_correlations(dc_df, tg_df)
        + validate_exchange_concentration(dc_df, tg_df)
        + validate_monthly_decay(dc_df, tg_df)
        + validate_post_pump(dc_df, tg_df)
        + validate_rank_buckets(dc_df, tg_df)
    )
    if reg_results:
        all_checks += validate_regression_signs(reg_results)
    for c in all_checks:
        report.add(c)
    return report
