"""
Synthetic Data Generator — Cryptocurrency Pump & Dump Ecosystem
================================================================
Calibrated to Hamrick et al. (2018/2019). All paper statistics targeted.

Approach:
  1. Draw rank from log-normal matching paper mean/std
  2. Assign rank-bucket label post-hoc for Table 3 compliance
  3. Draw price_inc so bucket medians match Table 3 AND overall mean ~ paper
  4. Apply exchange multiplier BEFORE monthly rescaling (so both effects preserved)
  5. Post-pump: beta mixture → median ≈ -38/-41%, pct_neg ≈ 62%
"""

import numpy as np
import pandas as pd
from typing import Tuple

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Paper-reported stats ──────────────────────────────────────────────────────
PAPER_STATS = {
    "discord": {
        "n": 952,
        "mean_price_inc": 6.78,  "std_price_inc": 17.34,
        "mean_exchanges":  21.11, "std_exchanges": 26.50,
        "mean_rank":      257.64, "std_rank":      309.30,
        "mean_pair_count": 24.74, "std_pair_count": 89.05,
        "mean_server_members": 5373, "std_server_members": 9467,
        "corr_rank_price": 0.46, "corr_exchanges_price": -0.15,
    },
    "telegram": {
        "n": 2469,
        "mean_price_inc": 9.57,  "std_price_inc": 22.93,
        "mean_exchanges":  17.72, "std_exchanges": 22.50,
        "mean_rank":      375.0,  "std_rank":      417.0,
        "mean_pair_count": 16.89, "std_pair_count": 64.17,
        "mean_views":     9649,   "std_views":      9815,
        "corr_rank_price": 0.40,  "corr_exchanges_price": -0.14,
    },
}

# Table 8 — exact monthly medians
MONTHLY_DECAY = {
    "discord":  {1: 5.4, 2: 4.1, 3: 3.9, 4: 3.2, 5: 2.9, 6: 2.2},
    "telegram": {1: 6.4, 2: 4.9, 3: 5.2, 4: 4.2, 5: 2.8, 6: 3.2},
}

# Table 3 — median price increase per rank bucket
RANK_BUCKET_MEDIANS = {
    "discord":  {"le75": 3.51, "76_200": 5.22, "201_500": 5.32, "gt500": 23.23},
    "telegram": {"le75": 4.81, "76_200": 6.46, "201_500": 8.10, "gt500": 18.74},
}

# Exchange share (Tables 1 & 2)
EXCHANGE_DIST = {
    "discord":  {"Binance": 0.22, "Bittrex": 0.20, "Binance+Bittrex": 0.08, "Other": 0.074, "None": 0.426},
    "telegram": {"Binance": 0.22, "Bittrex": 0.18, "Binance+Bittrex": 0.05, "Other": 0.07, "None": 0.48},
}

# Monthly share (Tables 1 & 2)
MONTHLY_DIST = {
    "discord":  {1: 0.15, 2: 0.12, 3: 0.13, 4: 0.12, 5: 0.37, 6: 0.11},
    "telegram": {1: 0.16, 2: 0.12, 3: 0.13, 4: 0.27, 5: 0.19, 6: 0.13},
}

# Exchange price multiplier (encodes Table 6 negative signs for Binance/Bittrex dummies)
EXCHANGE_PRICE_MULT = {
    "Binance":         0.72,
    "Bittrex":         0.82,
    "Binance+Bittrex": 0.63,
    "Other":           1.18,
    "None":            1.00,
}

CHANNEL_NAMES_TG = [
    "Big Pump Signal","Crypto4Pumps","CryptoPumpAlerts","PumpKingCrypto",
    "CoinPumpNation","PumpHunters","MoonPumpSignals","ElitePumpGroup",
    "CryptoCartelOriginal","PureInvestments","PumpMasterBTC","AlphaPumpLeague","CryptoRocketPumps",
]
CHANNEL_NAMES_DC = [
    "PumpStation#general","CryptoGainz#pumps","MoonMission#signals","PumpKings#alerts",
    "CoinRocket#trade","PumpElite#main","CryptoHunters#pump","GainersClub#signals",
    "PumpNation#discord","TradeSignals#crypto","MoonshotHQ#pumps","CryptoAlert#pump",
]


def _lognormal_params(mean: float, std: float) -> Tuple[float, float]:
    """Convert arithmetic mean & std → log-normal mu, sigma."""
    var  = std ** 2
    s2   = np.log(1 + var / mean**2)
    mu   = np.log(mean) - s2 / 2
    return mu, np.sqrt(s2)


def _rank_bucket(rank: np.ndarray) -> np.ndarray:
    """Map rank values → bucket labels."""
    labels = np.empty(len(rank), dtype=object)
    labels[rank <= 75]                      = "le75"
    labels[(rank > 75)  & (rank <= 200)]    = "76_200"
    labels[(rank > 200) & (rank <= 500)]    = "201_500"
    labels[rank > 500]                      = "gt500"
    return labels


def _draw_price_per_bucket(rank: np.ndarray, bucket_medians: dict, noise_sigma=0.8) -> np.ndarray:
    """
    Draw price_inc for each observation so each bucket's MEDIAN matches Table 3.
    Uses log-normal within each bucket (mu = log(target_median), sigma=noise_sigma).
    Forces exact median post-hoc via multiplicative scaling within bucket.
    """
    buckets = _rank_bucket(rank)
    price   = np.empty(len(rank))
    for label, target_med in bucket_medians.items():
        mask = buckets == label
        n    = mask.sum()
        if n == 0:
            continue
        raw = np.exp(rng.normal(np.log(target_med), noise_sigma, n))
        raw = np.clip(raw, 0.05, 600.0)
        # Force exact median
        raw *= target_med / np.median(raw)
        price[mask] = raw
    return price


def _assign(n, dist):
    return rng.choice(list(dist.keys()), size=n, p=list(dist.values()))


def _channels(n, names, top_share=0.45):
    n_top = 3
    top   = top_share / n_top
    rest  = (1 - top_share) / max(len(names) - n_top, 1)
    p     = np.array([top]*n_top + [rest]*(len(names)-n_top))
    p     = p[:len(names)]; p /= p.sum()
    return rng.choice(names, size=n, p=p)


def _post_pump(n: int) -> np.ndarray:
    """
    Paper: median ≈ -40%, ~62% negative.
    Beta mixture calibrated to these targets.
    """
    neg_mask = rng.random(n) < 0.62
    out = np.empty(n)
    n_neg = neg_mask.sum(); n_pos = n - n_neg
    b = rng.beta(2.8, 1.8, n_neg)
    out[neg_mask]  = -(b * 72 + 5)               # range [-77, -5]
    out[~neg_mask] = rng.uniform(-3, 18, n_pos)
    return np.clip(out, -95, 50)


def _build(n, stats, key, ch_names, ch_conc, pump_types, pump_probs):
    # 1. Draw rank from log-normal (matches paper mean/std)
    mu_r, sig_r = _lognormal_params(stats["mean_rank"], stats["std_rank"])
    rank = np.clip(np.exp(rng.normal(mu_r, sig_r, n)).astype(int), 2, 2036)

    # 2. Draw exchanges correlated with rank (r_re = -0.42)
    mu_e, sig_e = _lognormal_params(stats["mean_exchanges"], stats["std_exchanges"])
    z_r  = (np.log(rank.clip(2)) - mu_r) / sig_r
    z_e  = -0.42 * z_r + np.sqrt(1 - 0.42**2) * rng.standard_normal(n)
    exchanges = np.clip(np.exp(mu_e + sig_e * z_e).astype(int), 1, 182)

    # 3. Draw pair_count (positive correlation with exchanges, slight with price)
    mu_pc, sig_pc = _lognormal_params(stats["mean_pair_count"], stats["std_pair_count"])
    z_pc = 0.65 * (np.log(exchanges.clip(1)) - mu_e) / sig_e + \
           np.sqrt(1 - 0.65**2) * rng.standard_normal(n)
    pair_count = np.clip(np.exp(mu_pc + sig_pc * z_pc).astype(int), 1, 759)

    # 4. Draw price_inc per rank-bucket (Table 3 medians exactly)
    price_inc = _draw_price_per_bucket(rank, RANK_BUCKET_MEDIANS[key], noise_sigma=0.80)

    # 5. Apply exchange multiplier (Table 6 sign effects)
    months       = _assign(n, MONTHLY_DIST[key])
    exchange_col = _assign(n, EXCHANGE_DIST[key])
    exch_mult    = np.array([EXCHANGE_PRICE_MULT[e] for e in exchange_col])
    price_inc   *= exch_mult

    # Add positive pair_count partial effect (Table 6: ln_pair_count coef > 0)
    # Small multiplicative boost so partial corr is positive after controlling for rank
    pc_z = (np.log(pair_count.clip(1)) - np.log(pair_count.clip(1)).mean()) /            (np.log(pair_count.clip(1)).std() + 1e-9)
    price_inc *= np.exp(0.10 * pc_z)   # ~10% boost per SD of ln(pair_count)

    # Exchanges: higher exchanges = lower price (Table 6: ln_exchanges coef < 0)
    ex_z = (np.log(exchanges.clip(1)) - np.log(exchanges.clip(1)).mean()) /            (np.log(exchanges.clip(1)).std() + 1e-9)
    price_inc *= np.exp(-0.12 * ex_z)  # ~12% penalty per SD of ln(exchanges)

    # 6. Rescale within each month so medians EXACTLY match Table 8
    for m, target in MONTHLY_DECAY[key].items():
        mask = months == m
        if mask.sum() == 0:
            continue
        raw_med = np.median(price_inc[mask])
        if raw_med > 0:
            price_inc[mask] *= (target / raw_med)

    price_inc = np.clip(price_inc, 0.05, 342.0)

    # 7. Other columns
    channels  = _channels(n, ch_names, ch_conc)
    post_pump = _post_pump(n)
    p_types   = rng.choice(pump_types, n, p=pump_probs)

    return rank, exchanges, pair_count, months, exchange_col, channels, price_inc, post_pump, p_types


def generate_discord(n=None):
    stats = PAPER_STATS["discord"]
    n = n or stats["n"]

    rank, exchanges, pair_count, months, exchange_col, channels, price_inc, post_pump, p_types = \
        _build(n, stats, "discord", CHANNEL_NAMES_DC, 0.40,
               ["obvious","target","copied"], [0.11, 0.42, 0.47])

    mu_sm, sig_sm = _lognormal_params(stats["mean_server_members"], stats["std_server_members"])
    server_members = np.clip(rng.lognormal(mu_sm, sig_sm, n).astype(int), 141, 49415)

    df = pd.DataFrame({
        "platform": "Discord",
        "pump_id":              np.arange(1, n+1),
        "month":                months,
        "coin_rank":            rank,
        "n_exchanges":          exchanges,
        "pair_count":           pair_count,
        "channel":              channels,
        "exchange":             exchange_col,
        "max_price_inc_pct":    np.round(price_inc, 4),
        "post_pump_change_pct": np.round(post_pump, 4),
        "server_members":       server_members,
        "views":                0,
        "pump_type":            p_types,
    })
    df["date"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        (df["month"]-1)*30 + rng.integers(0, 30, n), unit="D")
    return df


def generate_telegram(n=None):
    stats = PAPER_STATS["telegram"]
    n = n or stats["n"]

    rank, exchanges, pair_count, months, exchange_col, channels, price_inc, post_pump, p_types = \
        _build(n, stats, "telegram", CHANNEL_NAMES_TG, 0.45,
               ["obvious","target"], [0.12, 0.88])

    mu_v, sig_v = _lognormal_params(stats["mean_views"], stats["std_views"])
    views = np.clip(rng.lognormal(mu_v, sig_v, n).astype(int), 0, 77266)

    df = pd.DataFrame({
        "platform": "Telegram",
        "pump_id":              np.arange(1, n+1),
        "month":                months,
        "coin_rank":            rank,
        "n_exchanges":          exchanges,
        "pair_count":           pair_count,
        "channel":              channels,
        "exchange":             exchange_col,
        "max_price_inc_pct":    np.round(price_inc, 4),
        "post_pump_change_pct": np.round(post_pump, 4),
        "server_members":       0,
        "views":                views,
        "pump_type":            p_types,
    })
    df["date"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        (df["month"]-1)*30 + rng.integers(0, 30, n), unit="D")
    return df


def generate_all(discord_n=None, telegram_n=None):
    return generate_discord(discord_n), generate_telegram(telegram_n)


if __name__ == "__main__":
    dc, tg = generate_all()
    print(f"Discord:  N={len(dc)} | mean={dc['max_price_inc_pct'].mean():.2f}% | "
          f"median={dc['max_price_inc_pct'].median():.2f}%  (paper 6.78 / 3.5)")
    print(f"Telegram: N={len(tg)} | mean={tg['max_price_inc_pct'].mean():.2f}% | "
          f"median={tg['max_price_inc_pct'].median():.2f}%  (paper 9.57 / 5.1)")
    print(f"Discord  rank mean: {dc['coin_rank'].mean():.1f}  (paper 257.6)")
    print(f"Telegram rank mean: {tg['coin_rank'].mean():.1f}  (paper 375.0)")
    print(f"Discord  exch mean: {dc['n_exchanges'].mean():.1f}  (paper 21.1)")
    print(f"Telegram exch mean: {tg['n_exchanges'].mean():.1f}  (paper 17.7)")
    for label, lo, hi in [("≤75",0,75),("76-200",76,200),("201-500",201,500),(">500",501,9999)]:
        m = dc[dc["coin_rank"].between(lo,hi)]["max_price_inc_pct"].median()
        print(f"  Discord rank {label}: median={m:.2f}%")
    print(f"Monthly DC: {dict(dc.groupby('month')['max_price_inc_pct'].median().round(2))}")
    print(f"Monthly TG: {dict(tg.groupby('month')['max_price_inc_pct'].median().round(2))}")
    print(f"Post-pump DC: median={dc['post_pump_change_pct'].median():.1f}%  "
          f"pct_neg={( dc['post_pump_change_pct']<0).mean()*100:.1f}%")
    print(f"Corr(rank,price) DC: {dc['coin_rank'].corr(dc['max_price_inc_pct']):.3f}  (paper 0.46)")
    print(f"Corr(rank,price) TG: {tg['coin_rank'].corr(tg['max_price_inc_pct']):.3f}  (paper 0.40)")

# ── post-generation calibration helper (used by tests) ───────────────────────
def get_calibration_summary(dc_df, tg_df):
    """Return dict of key statistics for both platforms."""
    return {
        "discord": {
            "n": len(dc_df),
            "mean_price_inc": dc_df["max_price_inc_pct"].mean(),
            "median_price_inc": dc_df["max_price_inc_pct"].median(),
            "mean_rank": dc_df["coin_rank"].mean(),
            "corr_rank_price": dc_df["coin_rank"].corr(dc_df["max_price_inc_pct"]),
            "monthly_medians": dc_df.groupby("month")["max_price_inc_pct"].median().to_dict(),
        },
        "telegram": {
            "n": len(tg_df),
            "mean_price_inc": tg_df["max_price_inc_pct"].mean(),
            "median_price_inc": tg_df["max_price_inc_pct"].median(),
            "mean_rank": tg_df["coin_rank"].mean(),
            "corr_rank_price": tg_df["coin_rank"].corr(tg_df["max_price_inc_pct"]),
            "monthly_medians": tg_df.groupby("month")["max_price_inc_pct"].median().to_dict(),
        },
    }
