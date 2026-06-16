"""HTML Tearsheet Generator — Hamrick et al. Replication Study."""

import base64, json
from pathlib import Path
from datetime import datetime
import pandas as pd


def _b64_img(path: Path) -> str:
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return ""


def generate_tearsheet(dc_df, tg_df, reg_results, eco_report, val_report, plot_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect metrics ───────────────────────────────────────────────────
    val_df   = val_report.to_dataframe()
    pass_rate = val_report.pass_rate

    dc_reg = reg_results["discord"]
    tg_reg = reg_results["telegram"]
    r2c    = reg_results["r2_comparison"]
    dc_pp  = reg_results["post_pump"]["discord"]
    tg_pp  = reg_results["post_pump"]["telegram"]

    dc_eco = eco_report["discord"]
    tg_eco = eco_report["telegram"]

    # Coefficient tables
    def coef_rows(res):
        rows = ""
        for var, info in res["sign_checks"].items():
            ok = "✓" if info["match"] else "✗"
            sig = "***" if info["pval"] < 0.01 else ("**" if info["pval"] < 0.05 else ("*" if info["pval"] < 0.10 else ""))
            color = "#3fb950" if info["match"] else "#f78166"
            rows += f'<tr><td>{var}</td><td>{info["coef"]:.4f}{sig}</td><td>{info["pval"]:.4f}</td><td style="color:{color}">{ok} {info["expected"]}</td></tr>'
        return rows

    # Validation table rows
    val_rows = ""
    for _, row in val_df.iterrows():
        color = "#3fb950" if row["passed"] == "PASS" else "#f78166"
        symbol = "✓" if row["passed"] == "PASS" else "✗"
        val_rows += (
            f'<tr><td>{row["section"]}</td><td>{row["check"]}</td>'
            f'<td>{row["paper_value"]}</td><td>{row["replicated_value"]}</td>'
            f'<td>{row["delta_pct"]:.1f}%</td><td style="color:{color}">{symbol} {row["passed"]}</td></tr>'
        )

    # Monthly profitability table
    from src.data.synthetic_generator import MONTHLY_DECAY
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun"]
    dc_monthly = dc_df.groupby("month")["max_price_inc_pct"].median().reindex(range(1,7))
    tg_monthly = tg_df.groupby("month")["max_price_inc_pct"].median().reindex(range(1,7))
    monthly_rows = ""
    for i, m in enumerate(range(1, 7)):
        dc_rep = dc_monthly.get(m, 0); tg_rep = tg_monthly.get(m, 0)
        dc_pap = MONTHLY_DECAY["discord"][m]; tg_pap = MONTHLY_DECAY["telegram"][m]
        monthly_rows += (
            f'<tr><td>{month_labels[i]}</td>'
            f'<td>{dc_pap}</td><td>{dc_rep:.2f}</td>'
            f'<td>{tg_pap}</td><td>{tg_rep:.2f}</td></tr>'
        )

    # Rank bucket table
    bins   = [0, 75, 200, 500, float("inf")]
    blabels = ["≤75", "76-200", "201-500", ">500"]
    paper_dc = [3.51, 5.22, 5.32, 23.23]
    paper_tg = [4.81, 6.46, 8.10, 18.74]
    dc_df2 = dc_df.copy(); dc_df2["rb"] = pd.cut(dc_df2["coin_rank"], bins=bins, labels=blabels)
    tg_df2 = tg_df.copy(); tg_df2["rb"] = pd.cut(tg_df2["coin_rank"], bins=bins, labels=blabels)
    bucket_rows = ""
    for i, lb in enumerate(blabels):
        dc_med = dc_df2.loc[dc_df2["rb"]==lb, "max_price_inc_pct"].median()
        tg_med = tg_df2.loc[tg_df2["rb"]==lb, "max_price_inc_pct"].median()
        bucket_rows += (
            f'<tr><td>{lb}</td><td>{paper_dc[i]}</td><td>{dc_med:.2f}</td>'
            f'<td>{paper_tg[i]}</td><td>{tg_med:.2f}</td></tr>'
        )

    # Encode images
    imgs = {k: _b64_img(v) for k, v in plot_paths.items()}

    def img_tag(key, caption):
        src = imgs.get(key, "")
        if not src:
            return f'<p style="color:#888">Figure not available</p>'
        return f'<figure><img src="{src}" style="width:100%;border-radius:8px"><figcaption>{caption}</figcaption></figure>'

    # ── HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto P&D Replication — Hamrick et al.</title>
<style>
  :root {{
    --bg:#0d1117; --panel:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e; --blue:#58a6ff;
    --green:#3fb950; --red:#f78166; --purple:#d2a8ff;
    --yellow:#e3b341;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ background:var(--bg); color:var(--text); font-family:'Courier New',monospace; font-size:14px; padding:24px }}
  h1 {{ font-size:22px; color:var(--blue); margin-bottom:4px }}
  h2 {{ font-size:16px; color:var(--purple); border-bottom:1px solid var(--border); padding-bottom:6px; margin:32px 0 16px }}
  h3 {{ font-size:14px; color:var(--blue); margin:16px 0 8px }}
  .subtitle {{ color:var(--muted); font-size:12px; margin-bottom:24px }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; margin:16px 0 }}
  .kpi {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:14px }}
  .kpi-label {{ font-size:11px; color:var(--muted); margin-bottom:4px }}
  .kpi-value {{ font-size:20px; font-weight:700 }}
  .kpi-sub {{ font-size:11px; color:var(--muted); margin-top:2px }}
  .blue {{ color:var(--blue) }} .green {{ color:var(--green) }} .red {{ color:var(--red) }} .purple {{ color:var(--purple) }} .yellow {{ color:var(--yellow) }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:18px; margin:16px 0 }}
  table {{ width:100%; border-collapse:collapse; font-size:12px }}
  th {{ background:var(--border); color:var(--muted); padding:8px 10px; text-align:left; font-weight:normal }}
  td {{ padding:7px 10px; border-bottom:1px solid var(--border) }}
  tr:hover td {{ background:rgba(255,255,255,0.03) }}
  .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:16px }}
  .progress-bar {{ background:var(--border); border-radius:4px; height:12px; overflow:hidden; margin:6px 0 }}
  .progress-fill {{ height:100%; border-radius:4px; transition:width 0.3s }}
  figcaption {{ font-size:11px; color:var(--muted); text-align:center; margin-top:6px; font-style:italic }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600 }}
  .badge-pass {{ background:rgba(63,185,80,0.2); color:var(--green) }}
  .badge-fail {{ background:rgba(247,129,102,0.2); color:var(--red) }}
  .section-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px }}
  @media(max-width:768px) {{ .two-col,.section-grid {{ grid-template-columns:1fr }} }}
</style>
</head>
<body>

<h1>🔍 Cryptocurrency Pump &amp; Dump Ecosystem</h1>
<div class="subtitle">
  Replication Study — Hamrick, Rouhi, Mukherjee, Feder, Gandal, Moore &amp; Vasek (2018/2019) &nbsp;|&nbsp;
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Discord Observations</div>
    <div class="kpi-value blue">{len(dc_df):,}</div>
    <div class="kpi-sub">Paper: 952</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram Observations</div>
    <div class="kpi-value green">{len(tg_df):,}</div>
    <div class="kpi-sub">Paper: 2,469</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Validation Pass Rate</div>
    <div class="kpi-value {'green' if pass_rate >= 70 else 'yellow'}">{pass_rate:.1f}%</div>
    <div class="kpi-sub">{val_report.n_passed}/{len(val_report.checks)} checks</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Discord Adj R²</div>
    <div class="kpi-value blue">{dc_reg['adj_r2']:.4f}</div>
    <div class="kpi-sub">Paper: 0.30</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram Adj R²</div>
    <div class="kpi-value green">{tg_reg['adj_r2']:.4f}</div>
    <div class="kpi-sub">Paper: 0.32</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Discord Median Post-Pump</div>
    <div class="kpi-value red">{dc_pp['median_post_pump_pct']:.1f}%</div>
    <div class="kpi-sub">Paper: −41%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram Median Post-Pump</div>
    <div class="kpi-value red">{tg_pp['median_post_pump_pct']:.1f}%</div>
    <div class="kpi-sub">Paper: −38%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Binance+Bittrex Share (TG)</div>
    <div class="kpi-value purple">{tg_eco['exchange_concentration']['binance_bittrex_combined_pct']:.1f}%</div>
    <div class="kpi-sub">Paper: ~86%</div>
  </div>
</div>

<!-- Section 1: Monthly Decay -->
<h2>§ Profitability Decay Over Time — Table 8 Replication</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:12px">
  Paper finding: median profitability fell ~50% (Telegram) and ~60% (Discord) from January to June 2018.
</p>
{img_tag("monthly_decay", "Figure 1: Monthly median pump profitability — replicated (solid) vs paper Table 8 (dashed)")}

<div class="panel">
  <table>
    <thead><tr><th>Month</th><th>Discord (Paper)</th><th>Discord (Rep.)</th><th>Telegram (Paper)</th><th>Telegram (Rep.)</th></tr></thead>
    <tbody>{monthly_rows}</tbody>
  </table>
</div>

<!-- Section 2: Rank Buckets -->
<h2>§ Coin Rank vs Pump Success — Table 3 Replication</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:12px">
  Key finding: median price increase for rank >500 coins is 5–6× higher than for top-75 coins.
</p>
<div class="section-grid">
  {img_tag("rank_vs_price", "Figure 2: Log-log scatter of coin rank vs price increase")}
  {img_tag("rank_buckets", "Figure 4: Box plots of price increase by rank bucket")}
</div>
<div class="panel">
  <table>
    <thead><tr><th>Rank Bucket</th><th>Discord Paper</th><th>Discord Rep.</th><th>Telegram Paper</th><th>Telegram Rep.</th></tr></thead>
    <tbody>{bucket_rows}</tbody>
  </table>
</div>

<!-- Section 3: Exchange Concentration -->
<h2>§ Exchange Concentration — Section 3.3</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:12px">
  Paper: Binance and Bittrex together account for 86–87% of exchange-listed pumps.
</p>
{img_tag("exchange_concentration", "Figure 3: Distribution of pumps across exchanges")}
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Discord B+B Share</div>
    <div class="kpi-value blue">{dc_eco['exchange_concentration']['binance_bittrex_combined_pct']:.1f}%</div>
    <div class="kpi-sub">Paper: ~87%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram B+B Share</div>
    <div class="kpi-value green">{tg_eco['exchange_concentration']['binance_bittrex_combined_pct']:.1f}%</div>
    <div class="kpi-sub">Paper: ~86%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Discord — No Exchange Listed</div>
    <div class="kpi-value blue">{dc_eco['exchange_concentration']['pct_no_exchange']:.1f}%</div>
    <div class="kpi-sub">Paper: ~46%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram — No Exchange Listed</div>
    <div class="kpi-value green">{tg_eco['exchange_concentration']['pct_no_exchange']:.1f}%</div>
    <div class="kpi-sub">Paper: ~48%</div>
  </div>
</div>

<!-- Section 4: Channel Concentration -->
<h2>§ Channel Concentration — Section 3.3</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:12px">
  Paper: 3 channels account for ~45% of all Telegram pumps (highly concentrated ecosystem).
</p>
{img_tag("channel_lorenz", "Figure 6: Lorenz curve of pump distribution across channels")}
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Discord Top-3 Channel Share</div>
    <div class="kpi-value blue">{dc_eco['channel_concentration']['top_n_share_pct']:.1f}%</div>
    <div class="kpi-sub">Paper (Telegram): ~45%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram Top-3 Channel Share</div>
    <div class="kpi-value green">{tg_eco['channel_concentration']['top_n_share_pct']:.1f}%</div>
    <div class="kpi-sub">Paper: ~45%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Discord HHI (Channels)</div>
    <div class="kpi-value blue">{dc_eco['channel_concentration']['hhi']:.4f}</div>
    <div class="kpi-sub">1.0 = monopoly</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram HHI (Channels)</div>
    <div class="kpi-value green">{tg_eco['channel_concentration']['hhi']:.4f}</div>
    <div class="kpi-sub">1.0 = monopoly</div>
  </div>
</div>

<!-- Section 5: Post-Pump Crash -->
<h2>§ Post-Pump Price Crash — Section 5.2</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:12px">
  Paper: median post-pump price change = −41% (Discord), −38% (Telegram) within 48 hours. >60% of coins end below pre-pump price.
</p>
{img_tag("post_pump", "Figure 5: Distribution of post-pump (48h) price changes")}
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-label">Discord % Below Pre-Pump</div>
    <div class="kpi-value red">{dc_pp['pct_below_pre_pump']:.1f}%</div>
    <div class="kpi-sub">Paper: >60%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Telegram % Below Pre-Pump</div>
    <div class="kpi-value red">{tg_pp['pct_below_pre_pump']:.1f}%</div>
    <div class="kpi-sub">Paper: >60%</div>
  </div>
</div>

<!-- Section 6: Regression Table -->
<h2>§ Regression Results — Table 6 Replication (Log/Log OLS)</h2>
<p style="color:var(--muted);font-size:12px;margin-bottom:12px">
  Dependent variable: ln(Max 5-min % price increase). Clustered SEs at coin level.
</p>
<div class="two-col">
  <div class="panel">
    <h3>Discord (Adj R² = {dc_reg['adj_r2']:.4f}, N={dc_reg['n_obs']:,})</h3>
    <table>
      <thead><tr><th>Variable</th><th>Coef.</th><th>p-val</th><th>Sign Check</th></tr></thead>
      <tbody>{coef_rows(dc_reg)}</tbody>
    </table>
  </div>
  <div class="panel">
    <h3>Telegram (Adj R² = {tg_reg['adj_r2']:.4f}, N={tg_reg['n_obs']:,})</h3>
    <table>
      <thead><tr><th>Variable</th><th>Coef.</th><th>p-val</th><th>Sign Check</th></tr></thead>
      <tbody>{coef_rows(tg_reg)}</tbody>
    </table>
  </div>
</div>

<!-- Section 7: Full Validation Report -->
<h2>§ Full Validation Report — Paper vs Replicated</h2>
<div class="panel">
  <div style="margin-bottom:12px">
    <span class="badge badge-pass">✓ {val_report.n_passed} PASS</span>
    &nbsp;
    <span class="badge badge-fail">✗ {val_report.n_failed} FAIL</span>
    &nbsp;
    <span style="color:var(--muted);font-size:12px">({pass_rate:.1f}% pass rate)</span>
  </div>
  <div class="progress-bar">
    <div class="progress-fill" style="width:{pass_rate:.0f}%;background:{'var(--green)' if pass_rate>=70 else 'var(--yellow)'}"></div>
  </div>
  <br>
  <table>
    <thead><tr><th>Section</th><th>Check</th><th>Paper</th><th>Replicated</th><th>Δ%</th><th>Result</th></tr></thead>
    <tbody>{val_rows}</tbody>
  </table>
</div>

<!-- Methodology Notes -->
<h2>§ Methodology &amp; Limitations</h2>
<div class="panel" style="color:var(--muted);font-size:12px;line-height:1.8">
  <p><strong style="color:var(--text)">Data:</strong>
    Synthetic dataset generated via calibrated multivariate log-normal distribution preserving paper's
    marginal moments (means, SDs), pairwise correlations (Tables 4–5), monthly distributions (Table 1–2),
    and exchange/channel concentration ratios (Section 3.3). Seed fixed at 42 for reproducibility.
  </p><br>
  <p><strong style="color:var(--text)">Regression:</strong>
    Log/log OLS matching Table 6 specification exactly. Clustered standard errors by coin-rank decile
    (proxy for coin identity, since we lack true coin IDs). Monthly dummies Jan=reference, same as paper.
  </p><br>
  <p><strong style="color:var(--text)">Limitations:</strong>
    (1) No real trading data — synthetic only, so absolute coefficient magnitudes differ from paper.
    (2) Coin clustering uses rank deciles rather than true coin IDs, slightly overstating SE precision.
    (3) Volume data (Section 5.3) not replicated — coinmarketcap.com 24h rolling volume not reproducible synthetically.
    (4) Google Trends Figure 1 (paper appendix) not replicated — requires live API call.
  </p><br>
  <p><strong style="color:var(--text)">Validation tolerance:</strong>
    Descriptive stats ±15–20%, correlations ±25–50%, monthly medians ±40% (wider due to bin size),
    regression sign checks binary pass/fail, exchange shares ±8%.
  </p>
</div>

<div style="text-align:center;color:var(--muted);font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid var(--border)">
  Replication of: Hamrick et al. "An examination of the cryptocurrency pump and dump ecosystem" (2018/2019) &nbsp;|&nbsp;
  github.com/{'{tashrifulkabir34-lang}'}/crypto-pump-dump-replication
</div>

</body>
</html>"""

    out_path = out_dir / "tearsheet.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
