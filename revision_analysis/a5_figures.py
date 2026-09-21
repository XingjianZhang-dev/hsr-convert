"""A5: figures for the revision analyses, in the package's publication style."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from revision_analysis.common import FIG_DIR, RESULTS_DIR, ROOT, ensure_results_dirs

sys.path.insert(0, str(ROOT / "scripts"))
from round3_publication_assets import PROFILE_COLORS, setup_style  # noqa: E402

MODEL_LABELS = {
    "intercept_only": "Intercept only",
    "linear": "Linear",
    "ridge": "Ridge (primary)",
    "huber": "Huber",
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=400)
    plt.close(fig)


def fig_design_sensitivity() -> None:
    grid = pd.read_csv(RESULTS_DIR / "sensitivity_thresholds_grid.csv")
    main = grid[grid.burden_pct.eq(50)].pivot(index="eff_pct", columns="cap_pct", values="share_countries_same_as_baseline")
    draws = pd.read_csv(RESULTS_DIR / "sensitivity_weights_draws.csv")
    per_country = pd.read_csv(RESULTS_DIR / "sensitivity_thresholds.csv")

    fig = plt.figure(figsize=(7.15, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.5, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    im = ax_a.imshow(main.values, cmap="YlGnBu", vmin=0.80, vmax=1.0, origin="lower")
    ax_a.set_xticks(range(len(main.columns)), [f"{c}" for c in main.columns])
    ax_a.set_yticks(range(len(main.index)), [f"{r}" for r in main.index])
    ax_a.set_xlabel("Capacity cut-point (percentile of $C_i$)")
    ax_a.set_ylabel("Conversion cut-point (pct. of $E_i$)")
    ax_a.set_title("A. Labels unchanged vs. median rule", pad=5)
    for i in range(main.shape[0]):
        for j in range(main.shape[1]):
            v = main.values[i, j]
            ax_a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6, color="white" if v >= 0.95 else "#111111")
    fig.colorbar(im, ax=ax_a, shrink=0.8, pad=0.02)

    order = ["effective converters", "capacity under-realizers", "adaptive over-performers", "structurally vulnerable systems", "uncertain / data-limited systems"]
    short = {"effective converters": "Effective", "capacity under-realizers": "Under-realizer", "adaptive over-performers": "Over-performer", "structurally vulnerable systems": "Vulnerable", "uncertain / data-limited systems": "Uncertain"}
    bins = [0, 0.5, 0.7, 0.9, 0.999, 1.001]
    labels_b = ["<0.5", "0.5-0.7", "0.7-0.9", "0.9-<1", "1.0"]
    per_country["bin"] = pd.cut(per_country["stability_25"], bins=bins, labels=labels_b, right=False)
    ct = pd.crosstab(per_country["bin"], per_country["conversion_profile"]).reindex(columns=order, fill_value=0).reindex(labels_b, fill_value=0)
    bottom = np.zeros(len(ct))
    for p in order:
        ax_b.bar(range(len(ct)), ct[p].values, bottom=bottom, color=PROFILE_COLORS[p], label=short[p], width=0.72)
        bottom += ct[p].values
    ax_b.set_xticks(range(len(ct)), labels_b)
    ax_b.set_xlabel("Stability across 25 threshold combinations")
    ax_b.set_ylabel("Countries")
    ax_b.set_title("B. Per-country threshold stability", pad=5)
    ax_b.legend(frameon=True, fontsize=6.2, borderpad=0.3, handlelength=1.2)
    ax_b.grid(True, axis="y")

    schemes = [("dirichlet_moderate", "Moderate weight\nperturbation"), ("block_only_flat", "Flat block weights,\nequal within-block"), ("dirichlet_flat", "Flat block and\nwithin-block weights")]
    data_a = [draws[draws.scheme.eq(s)]["profile_agreement_with_baseline"].to_numpy() for s, _ in schemes]
    data_s = [draws[draws.scheme.eq(s)]["spearman_efficiency_vs_baseline"].to_numpy() for s, _ in schemes]
    pos = np.arange(len(schemes))
    bp1 = ax_c.boxplot(data_a, positions=pos - 0.17, widths=0.28, patch_artist=True, showfliers=False, medianprops={"color": "#111111"})
    bp2 = ax_c.boxplot(data_s, positions=pos + 0.17, widths=0.28, patch_artist=True, showfliers=False, medianprops={"color": "#111111"})
    for b in bp1["boxes"]:
        b.set(facecolor="#4c78a8", alpha=0.85)
    for b in bp2["boxes"]:
        b.set(facecolor="#f58518", alpha=0.85)
    ax_c.set_xticks(pos, [l for _, l in schemes])
    ax_c.set_ylim(0.6, 1.02)
    ax_c.set_ylabel("Agreement with baseline")
    ax_c.set_title("C. Weight perturbation (500 draws per scheme)", pad=5)
    ax_c.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Profile agreement", "Spearman of $E_i$"], frameon=True, loc="lower left", borderpad=0.35, handlelength=1.4)
    ax_c.grid(True, axis="y")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.11)
    _save(fig, "fig6_design_sensitivity")


def fig_calibration_models() -> None:
    cal = pd.read_csv(RESULTS_DIR / "calibration.csv")
    stat = json.loads((RESULTS_DIR / "stat_tests.json").read_text())["calibration"]
    mc = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    fig = plt.figure(figsize=(7.15, 3.5))
    ax_a = fig.add_axes([0.08, 0.16, 0.40, 0.74])
    ax_b = fig.add_axes([0.62, 0.16, 0.36, 0.74])

    x = cal["oof_predicted_ridge_mean_over_repeats"].to_numpy()
    y = cal["observed_shock_burden"].to_numpy()
    lo = cal["pi95_low"].to_numpy(); hi = cal["pi95_high"].to_numpy()
    xr = cal["oof_predicted_ridge_repeat1"].to_numpy()
    ax_a.vlines(xr, lo, hi, color="#bbbbbb", lw=0.6, alpha=0.7, zorder=1)
    ax_a.scatter(x, y, s=16, color="#386cb0", edgecolor="#111111", linewidth=0.3, alpha=0.9, zorder=3)
    lim = [min(x.min(), y.min()) - 300, max(x.max(), y.max()) + 300]
    ax_a.plot(lim, lim, color="#333333", lw=0.8, ls="--", label="Identity")
    xx = np.linspace(lim[0], lim[1], 50)
    ax_a.plot(xx, stat["calibration_intercept"] + stat["calibration_slope"] * xx, color="#d95f02", lw=1.1, label=f"Calibration fit (slope {stat['calibration_slope']:.2f})")
    ax_a.set_xlim(lim); ax_a.set_ylim(lim)
    ax_a.set_xlabel("Out-of-fold predicted excess deaths per million")
    ax_a.set_ylabel("Observed excess deaths per million")
    ax_a.set_title("A. Ridge calibration (n=106)", pad=5)
    ax_a.legend(frameon=True, loc="lower right", borderpad=0.35, handlelength=1.6)
    ax_a.text(0.03, 0.97, f"95% interval coverage {stat['coverage_95']:.2f}\n80% interval coverage {stat['coverage_80']:.2f}", transform=ax_a.transAxes, ha="left", va="top", fontsize=6.8, color="#333333")
    ax_a.grid(True)

    order = ["intercept_only", "elastic_net", "linear", "huber", "ridge", "gradient_boosting", "random_forest"]
    sub = mc.set_index("model").loc[order]
    ypos = np.arange(len(order))
    colors = ["#d95f02" if m == "ridge" else "#4c78a8" for m in order]
    ax_b.errorbar(sub["cv_rmse_mean"], ypos, xerr=sub["cv_rmse_sd"], fmt="none", ecolor="#777777", elinewidth=0.9, capsize=2)
    ax_b.scatter(sub["cv_rmse_mean"], ypos, color=colors, edgecolor="#111111", s=30, zorder=3)
    ax_b.set_yticks(ypos, [MODEL_LABELS[m] for m in order])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Repeated-CV RMSE (mean $\\pm$ SD over 100 folds)")
    ax_b.set_title("B. Expected-burden models", pad=5)
    ax_b.grid(True, axis="x")
    _save(fig, "fig7_calibration_and_models")


def supp_fig_baselines() -> None:
    dis = pd.read_csv(RESULTS_DIR / "disagreement_cases.csv")
    comp = pd.read_csv(RESULTS_DIR / "baseline_comparison.csv").set_index("baseline")
    prof = pd.read_csv(ROOT / "results/tables/conversion_profiles.csv")
    ghs = pd.read_csv(ROOT / "results/tables/stage4_preparedness_block_with_ghs.csv")[["iso3", "ghs_overall_2019"]]
    df = prof.merge(ghs, on="iso3", how="left")
    fig = plt.figure(figsize=(7.15, 3.4))
    ax_a = fig.add_axes([0.08, 0.16, 0.40, 0.74])
    ax_b = fig.add_axes([0.58, 0.16, 0.40, 0.74])
    for ax, col, xlabel, key in [(ax_a, "ghs_overall_2019", "GHS 2019 overall score", "ghs_2019_overall"), (ax_b, "potential_capacity_score", "Potential capacity $C_i$", "capacity_only")]:
        sub = df.dropna(subset=[col])
        ax.scatter(sub[col], sub["conversion_efficiency_zscore"], c=[PROFILE_COLORS.get(p, "#777777") for p in sub["conversion_profile"]], s=18, edgecolor="#111111", linewidth=0.3, alpha=0.9)
        ax.axhline(0, color="#333333", lw=0.7, ls="--")
        r = comp.loc[key]
        ax.set_title(f"{'A' if ax is ax_a else 'B'}. Spearman {r['spearman_with_conversion_efficiency']:.2f} [{r['spearman_ci_low']:.2f}, {r['spearman_ci_high']:.2f}], n={int(r['n'])}", pad=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Conversion efficiency $E_i$")
        ax.grid(True)
        named = dis[dis.baseline.eq(key)].sort_values("abs_rank_gap", ascending=False).head(6)
        for row in named.itertuples():
            src = sub[sub.iso3.eq(row.iso3)]
            if src.empty:
                continue
            ax.annotate(row.iso3, (src[col].iloc[0], src["conversion_efficiency_zscore"].iloc[0]), xytext=(3, 3), textcoords="offset points", fontsize=6.4, color="#222222")
    _save(fig, "supp_fig6_static_baselines")


def supp_fig_mc_settings() -> None:
    mc = pd.read_csv(RESULTS_DIR / "sensitivity_mc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9))
    for ax, col, ylabel in [(axes[0], "robust_profile_share", "Share of robust profiles (stability $\\geq$ 0.75)"), (axes[1], "most_likely_profile_agreement_with_baseline", "Most-likely profile = baseline label")]:
        for n_sim, sub in mc.groupby("n_simulations"):
            ax.plot(sub["noise_multiplier"], sub[col], marker="o", ms=4, label=f"{n_sim} draws")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2], ["0.5x", "1x", "2x"])
        ax.set_xlabel("Perturbation magnitude (multiple of default)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axes[0].set_ylim(0.5, 1.0); axes[1].set_ylim(0.9, 1.0)
    axes[0].set_title("A. Robust-profile share", pad=5); axes[1].set_title("B. Agreement with baseline labels", pad=5)
    axes[0].legend(frameon=True, fontsize=6.5, borderpad=0.3)
    fig.tight_layout()
    _save(fig, "supp_fig7_monte_carlo_settings")


def run() -> None:
    ensure_results_dirs()
    setup_style()
    fig_design_sensitivity()
    fig_calibration_models()
    supp_fig_baselines()
    supp_fig_mc_settings()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    run()
