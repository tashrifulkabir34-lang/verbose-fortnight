"""
Cryptocurrency Pump & Dump Ecosystem — Replication Study
=========================================================
Hamrick, Rouhi, Mukherjee, Feder, Gandal, Moore & Vasek (2018/2019)

Usage:
    python main.py
    python main.py --no-tearsheet
"""

import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_all
from src.analysis.regression import run_all_regressions, extract_coef_table
from src.analysis.ecosystem import full_ecosystem_report
from src.analysis.validation import run_full_validation
from src.visualization.plots import generate_all_plots


def banner(title):
    print(f"\n{'─'*70}\n  {title}\n{'─'*70}")


def run_pipeline(args):
    print("\n" + "="*70)
    print("  CRYPTO PUMP & DUMP REPLICATION — Hamrick et al. (2018/2019)")
    print("="*70)
    t0 = time.time()

    banner("1. Generating Synthetic Data")
    dc_df, tg_df = generate_all()
    print(f"  Discord:  {len(dc_df):,} obs | mean price inc: {dc_df['max_price_inc_pct'].mean():.2f}% (paper: 6.78%)")
    print(f"  Telegram: {len(tg_df):,} obs | mean price inc: {tg_df['max_price_inc_pct'].mean():.2f}% (paper: 9.57%)")

    banner("2. Log/Log OLS Regressions (Table 6 replication)")
    reg_results = run_all_regressions(dc_df, tg_df)
    for platform, res in [("Discord", reg_results["discord"]), ("Telegram", reg_results["telegram"])]:
        paper_r2 = 0.30 if platform == "Discord" else 0.32
        sign_ok = sum(1 for v in res["sign_checks"].values() if v["match"])
        sign_total = len(res["sign_checks"])
        print(f"  {platform}: Adj R²={res['adj_r2']:.4f} (paper={paper_r2}) | Signs {sign_ok}/{sign_total} match")

    banner("3. Ecosystem Concentration Analysis (Section 3.3)")
    eco = full_ecosystem_report(dc_df, tg_df)
    for platform, key in [("Discord", "discord"), ("Telegram", "telegram")]:
        cc = eco[key]["channel_concentration"]
        ec = eco[key]["exchange_concentration"]
        cr = eco[key]["coin_repetition"]
        pd_info = eco[key]["profitability_decline"]
        print(f"  {platform}: Top-3 channels={cc['top_n_share_pct']:.1f}% | "
              f"Binance+Bittrex={ec['binance_bittrex_combined_pct']:.1f}% | "
              f"Decline Jan→Jun={pd_info.get('decline_pct', 0):.1f}%")

    banner("4. Post-Pump Crash Analysis (Section 5.2)")
    for platform, key in [("Discord", "discord"), ("Telegram", "telegram")]:
        pp = reg_results["post_pump"][key]
        print(f"  {platform}: median={pp['median_post_pump_pct']:.1f}% (paper={pp['paper_median']:.0f}%) | "
              f"{pp['pct_below_pre_pump']:.1f}% below pre-pump (paper: >60%)")

    banner("5. Validation Against Paper Statistics")
    val_report = run_full_validation(dc_df, tg_df, reg_results)
    print(f"  {val_report.n_passed}/{len(val_report.checks)} checks passed ({val_report.pass_rate:.1f}%)")
    for c in val_report.checks:
        status = "✓" if c.passed else "✗"
        print(f"    {status} {c.name}: paper={c.paper_value}, got={c.replicated_value:.3f} (Δ={c.delta_pct:.1f}%)")

    banner("6. Generating Figures")
    out_dir = ROOT / "reports" / "figures"
    plot_paths = generate_all_plots(dc_df, tg_df, out_dir)
    for name, path in plot_paths.items():
        print(f"  ✓ {path.name}")

    banner("7. Saving CSVs")
    data_dir = ROOT / "reports" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dc_df.to_csv(data_dir / "discord_pumps.csv", index=False)
    tg_df.to_csv(data_dir / "telegram_pumps.csv", index=False)
    val_report.to_dataframe().to_csv(data_dir / "validation_report.csv", index=False)
    for p, res in [("discord", reg_results["discord"]), ("telegram", reg_results["telegram"])]:
        extract_coef_table(res).to_csv(data_dir / f"regression_{p}.csv")
    print("  ✓ discord_pumps.csv, telegram_pumps.csv, validation_report.csv, regression_*.csv")

    if not args.no_tearsheet:
        banner("8. HTML Tearsheet")
        from src.visualization.tearsheet import generate_tearsheet
        tp = generate_tearsheet(dc_df, tg_df, reg_results, eco, val_report, plot_paths, ROOT / "reports")
        print(f"  ✓ {tp}")

    print(f"\n{'='*70}")
    print(f"  Done in {time.time()-t0:.1f}s | Validation: {val_report.pass_rate:.1f}% pass rate")
    print(f"{'='*70}\n")
    return {"discord_df": dc_df, "telegram_df": tg_df, "reg": reg_results, "eco": eco, "val": val_report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tearsheet", action="store_true")
    return run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
