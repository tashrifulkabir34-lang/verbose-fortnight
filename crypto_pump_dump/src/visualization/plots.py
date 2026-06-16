"""Visualization — Replicated Figures for Hamrick et al."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
ACCENT1  = "#58A6FF"
ACCENT2  = "#3FB950"
ACCENT3  = "#F78166"
ACCENT4  = "#D2A8FF"
TEXT_COL = "#E6EDF3"
GRID_COL = "#30363D"
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
PAPER_MONTHLY_DC = [5.4, 4.1, 3.9, 3.2, 2.9, 2.2]
PAPER_MONTHLY_TG = [6.4, 4.9, 5.2, 4.2, 2.8, 3.2]


def _style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COL, "axes.labelcolor": TEXT_COL,
        "axes.titlecolor": TEXT_COL, "xtick.color": TEXT_COL,
        "ytick.color": TEXT_COL, "grid.color": GRID_COL, "grid.alpha": 0.5,
        "text.color": TEXT_COL, "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID_COL, "font.family": "monospace", "font.size": 11,
    })


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)


def plot_monthly_decay(dc_df, tg_df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pump Profitability Decay — Replication of Table 8", fontsize=14, color=TEXT_COL)
    for ax, df, platform, color, paper in [
        (axes[0], dc_df, "Discord", ACCENT1, PAPER_MONTHLY_DC),
        (axes[1], tg_df, "Telegram", ACCENT2, PAPER_MONTHLY_TG),
    ]:
        monthly = df.groupby("month")["max_price_inc_pct"].median().reindex(range(1, 7))
        x = np.arange(1, 7)
        ax.plot(x, paper, "--o", color=ACCENT3, label="Paper (Table 8)", lw=2, ms=7)
        ax.plot(x, monthly.values, "-o", color=color, label="Replicated", lw=2, ms=7)
        ax.fill_between(x, paper, monthly.values, alpha=0.15, color=ACCENT4)
        ax.set_xticks(x); ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel("Month"); ax.set_ylabel("Median % Price Increase")
        ax.set_title(platform); ax.legend(); ax.grid(True, alpha=0.4); ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = out_dir / "fig1_monthly_decay.png"; _save(fig, p); return p


def plot_rank_vs_price(dc_df, tg_df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Coin Rank vs Price Increase — Key Driver of Pump Success", fontsize=14, color=TEXT_COL)
    for ax, df, platform, color, paper_corr in [
        (axes[0], dc_df, "Discord", ACCENT1, 0.46),
        (axes[1], tg_df, "Telegram", ACCENT2, 0.40),
    ]:
        sample = df.sample(min(500, len(df)), random_state=42)
        ax.scatter(sample["coin_rank"], sample["max_price_inc_pct"], alpha=0.4, s=18, color=color)
        mask = (sample["coin_rank"] > 0) & (sample["max_price_inc_pct"] > 0)
        lx = np.log(sample.loc[mask, "coin_rank"]); ly = np.log(sample.loc[mask, "max_price_inc_pct"])
        z = np.polyfit(lx, ly, 1)
        xr = np.linspace(sample["coin_rank"].min(), sample["coin_rank"].max(), 200)
        ax.plot(xr, np.exp(z[1]) * xr ** z[0], color=ACCENT3, lw=2, label=f"Log-log fit")
        corr = df["coin_rank"].corr(df["max_price_inc_pct"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Coin Rank"); ax.set_ylabel("Max 5-min % Price Increase")
        ax.set_title(f"{platform} | r={corr:.3f} (paper: {paper_corr})")
        ax.legend(); ax.grid(True, alpha=0.4, which="both")
    fig.tight_layout()
    p = out_dir / "fig2_rank_vs_price.png"; _save(fig, p); return p


def plot_exchange_concentration(dc_df, tg_df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Exchange Concentration in Pump & Dump Schemes — Section 3.3", fontsize=14, color=TEXT_COL)
    colors_map = {"Binance": "#F0B90B", "Bittrex": ACCENT1, "Binance+Bittrex": "#FF6B35", "Other": ACCENT4, "None": GRID_COL}
    for ax, df, platform in [(axes[0], dc_df, "Discord"), (axes[1], tg_df, "Telegram")]:
        counts = df["exchange"].value_counts()
        labels = counts.index.tolist(); sizes = counts.values
        colors = [colors_map.get(l, "#888") for l in labels]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors,
            startangle=90, wedgeprops=dict(edgecolor=DARK_BG, linewidth=1.5), textprops=dict(color=TEXT_COL))
        [at.set_fontsize(9) for at in autotexts]
        ax.set_title(platform, fontsize=12)
    fig.tight_layout()
    p = out_dir / "fig3_exchange_concentration.png"; _save(fig, p); return p


def plot_rank_bucket_boxplot(dc_df, tg_df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Price Increase by Coin Rank Bucket — Replication of Table 3", fontsize=14, color=TEXT_COL)
    bins = [0, 75, 200, 500, np.inf]
    labels = ["≤75\n(Top)", "76–200", "201–500", ">500\n(Obscure)"]
    paper_dc = [3.51, 5.22, 5.32, 23.23]; paper_tg = [4.81, 6.46, 8.10, 18.74]
    for ax, df, platform, color, paper_vals in [
        (axes[0], dc_df, "Discord", ACCENT1, paper_dc),
        (axes[1], tg_df, "Telegram", ACCENT2, paper_tg),
    ]:
        df = df.copy()
        df["rb"] = pd.cut(df["coin_rank"], bins=bins, labels=labels)
        groups = [df.loc[df["rb"] == lb, "max_price_inc_pct"].dropna().values for lb in labels]
        bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color=ACCENT3, lw=2),
                        flierprops=dict(marker=".", color=color, alpha=0.3, ms=3))
        [p.set(facecolor=color, alpha=0.5) for p in bp["boxes"]]
        ax.plot(np.arange(1, 5), paper_vals, "D--", color=ACCENT3, ms=8, lw=2, label="Paper median")
        ax.set_xticklabels(labels); ax.set_xlabel("Coin Rank Bucket")
        ax.set_ylabel("Max % Price Increase"); ax.set_title(platform)
        ax.set_yscale("log"); ax.grid(True, alpha=0.4, which="both"); ax.legend()
    fig.tight_layout()
    p = out_dir / "fig4_rank_bucket_boxplot.png"; _save(fig, p); return p


def plot_post_pump_distribution(dc_df, tg_df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Post-Pump Price Changes (48h after pump) — Section 5.2", fontsize=14, color=TEXT_COL)
    paper_medians = {"Discord": -41.0, "Telegram": -38.0}
    for ax, df, platform, color in [
        (axes[0], dc_df, "Discord", ACCENT1),
        (axes[1], tg_df, "Telegram", ACCENT2),
    ]:
        post = df["post_pump_change_pct"].clip(-95, 50)
        ax.hist(post, bins=50, color=color, alpha=0.7, edgecolor=DARK_BG, linewidth=0.5)
        med = post.median(); paper_med = paper_medians[platform]
        ax.axvline(med, color=color, lw=2, label=f"Replicated median: {med:.1f}%")
        ax.axvline(paper_med, color=ACCENT3, lw=2, ls="--", label=f"Paper median: {paper_med}%")
        ax.axvline(0, color=TEXT_COL, lw=1, ls=":", alpha=0.5)
        neg_pct = (post < 0).mean() * 100
        ax.set_xlabel("Post-pump % Change"); ax.set_ylabel("Count")
        ax.set_title(f"{platform} | {neg_pct:.1f}% below pre-pump (paper: >60%)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    p = out_dir / "fig5_post_pump_distribution.png"; _save(fig, p); return p


def plot_channel_lorenz(dc_df, tg_df, out_dir):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Channel Concentration — Lorenz Curve", fontsize=14, color=TEXT_COL)
    for ax, df, platform, color in [
        (axes[0], dc_df, "Discord", ACCENT1),
        (axes[1], tg_df, "Telegram", ACCENT2),
    ]:
        counts = df["channel"].value_counts().sort_values()
        cumulative = np.cumsum(counts.values) / counts.sum()
        n = len(counts); x = np.linspace(0, 1, n)
        ax.plot(x, cumulative, color=color, lw=2, label="Lorenz curve")
        ax.plot([0, 1], [0, 1], "--", color=TEXT_COL, alpha=0.5, lw=1.5, label="Perfect equality")
        ax.fill_between(x, x, cumulative, alpha=0.2, color=color)
        top3_share = counts.tail(3).sum() / counts.sum()
        hhi = ((counts / counts.sum()) ** 2).sum()
        ax.set_xlabel("Cumulative share of channels"); ax.set_ylabel("Cumulative share of pumps")
        ax.set_title(f"{platform} | HHI={hhi:.3f} | Top-3 share={top3_share*100:.1f}%")
        ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    p = out_dir / "fig6_channel_lorenz.png"; _save(fig, p); return p


def generate_all_plots(dc_df, tg_df, out_dir):
    out_dir = Path(out_dir)
    return {
        "monthly_decay": plot_monthly_decay(dc_df, tg_df, out_dir),
        "rank_vs_price": plot_rank_vs_price(dc_df, tg_df, out_dir),
        "exchange_concentration": plot_exchange_concentration(dc_df, tg_df, out_dir),
        "rank_buckets": plot_rank_bucket_boxplot(dc_df, tg_df, out_dir),
        "post_pump": plot_post_pump_distribution(dc_df, tg_df, out_dir),
        "channel_lorenz": plot_channel_lorenz(dc_df, tg_df, out_dir),
    }
