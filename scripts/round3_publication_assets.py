from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
DATA = ROOT / "data" / "processed"
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"
TABLE_DIR = PAPER / "tables"
ASSET_FIG_DIR = ROOT / "paper_assets" / "figures"


PROFILE_COLORS = {
    "adaptive over-performers": "#1b9e77",
    "capacity under-realizers": "#d95f02",
    "effective converters": "#386cb0",
    "structurally vulnerable systems": "#7570b3",
    "uncertain / data-limited systems": "#8c8c8c",
}

LABEL_MARKERS = {
    "stable": "o",
    "evidence-dependent": "^",
    "data-limited": "s",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.35,
            "grid.color": "#b8b8b8",
            "grid.alpha": 0.26,
            "lines.linewidth": 1.1,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (FIG_DIR, ASSET_FIG_DIR):
        fig.savefig(directory / f"{name}.pdf")
        fig.savefig(directory / f"{name}.png", dpi=400)
        try:
            fig.savefig(directory / f"{name}.tiff", dpi=400)
        except Exception:
            pass
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def wrapped_label(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def label_group(category: str) -> str:
    if isinstance(category, str) and category.startswith("stable"):
        return "stable"
    if category == "evidence-dependent label":
        return "evidence-dependent"
    return "data-limited"


def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 4.85), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    layers = [
        (
            "A. Evidence layer",
            [
                "Pre-shock\nindicators\n2015-2019",
                "Feature\nblocks",
                "GHS 2019\npreparedness",
                "Validation\noutcomes",
            ],
            "#e8f2f1",
            "#0b6b5e",
        ),
        (
            "B. Conversion modeling layer",
            [
                "Potential\ncapacity",
                "Expected\nburden\nridge model",
                "Realized\ngap",
                "Conversion\nefficiency",
            ],
            "#f7efe4",
            "#9c5b12",
        ),
        (
            "C. Decision-support output layer",
            [
                "Conversion\nprofiles",
                "Bottleneck\nreview",
                "Label triage\nstable / evidence-\ndependent / limited",
                "Audit\npackage",
            ],
            "#eef0f8",
            "#404a8c",
        ),
    ]

    x0, w = 0.045, 0.91
    y_positions = [0.700, 0.395, 0.090]
    layer_h = 0.220
    box_w = 0.198
    gap = 0.026

    for (title, boxes, bg, edge), y in zip(layers, y_positions):
        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (x0, y),
                w,
                layer_h,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                facecolor=bg,
                edgecolor=edge,
                linewidth=0.95,
            )
        )
        ax.text(
            x0 + 0.017,
            y + layer_h - 0.030,
            title,
            weight="bold",
            color=edge,
            va="top",
            fontsize=8.7,
        )
        for j, box in enumerate(boxes):
            bx = x0 + 0.025 + j * (box_w + gap)
            by = y + 0.040
            ax.add_patch(
                mpl.patches.FancyBboxPatch(
                    (bx, by),
                    box_w,
                    0.108,
                    boxstyle="round,pad=0.009,rounding_size=0.010",
                    facecolor="white",
                    edgecolor=edge,
                    linewidth=0.72,
                )
            )
            ax.text(
                bx + box_w / 2,
                by + 0.054,
                box,
                ha="center",
                va="center",
                color="#222222",
                fontsize=7.9,
                linespacing=1.05,
            )

    for y1, y2 in [(y_positions[0], y_positions[1]), (y_positions[1], y_positions[2])]:
        ax.annotate(
            "",
            xy=(0.50, y2 + layer_h + 0.005),
            xytext=(0.50, y1 - 0.005),
            arrowprops={"arrowstyle": "->", "lw": 0.95, "color": "#555555"},
        )

    ax.text(
        0.04,
        0.025,
        "Guardrail: pandemic outcomes validate conversion; they do not enter pre-shock capacity scoring.",
        color="#444444",
        fontsize=7.4,
    )
    save_figure(fig, "fig1_hsr_convert_framework")


def fig2_capacity_vs_burden() -> None:
    profiles = read_csv("results/tables/conversion_profiles.csv")
    labels = read_csv("results/tables/stage4_country_label_stability.csv")
    df = profiles.merge(labels[["iso3", "country_label_category"]], on="iso3", how="left")
    df["label_group"] = df["country_label_category"].map(label_group)

    fig, ax = plt.subplots(figsize=(7.15, 4.75), constrained_layout=True)
    for group, marker in LABEL_MARKERS.items():
        sub = df[df["label_group"] == group]
        edge = "#111111" if group == "stable" else "#666666"
        ax.scatter(
            sub["potential_capacity_score"],
            sub["observed_shock_burden"],
            c=sub["conversion_efficiency_zscore"],
            cmap="RdYlGn",
            vmin=-2.5,
            vmax=2.5,
            s=38,
            marker=marker,
            edgecolor=edge,
            linewidth=0.58,
            alpha=0.88,
            label=f"{group} (n={len(sub)})",
        )

    # Annotated examples must be stable labels in the current run; non-stable candidates are skipped.
    for iso, dx, dy in [
        ("BLR", 0.008, -380),
        ("USA", 0.006, 220),
        ("DOM", 0.006, -420),
        ("DNK", 0.006, -350),
        ("SUR", 0.006, 180),
    ]:
        row = df[(df["iso3"] == iso) & df["country_label_category"].fillna("").str.startswith("stable")]
        if not row.empty:
            r = row.iloc[0]
            ax.annotate(
                iso,
                (r["potential_capacity_score"], r["observed_shock_burden"]),
                xytext=(r["potential_capacity_score"] + dx, r["observed_shock_burden"] + dy),
                arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#555555"},
                fontsize=6.9,
                color="#222222",
            )

    ax.set_xlabel("Potential capacity score")
    ax.set_ylabel("Observed shock burden\n(excess deaths per million)")
    ax.set_title("Capacity and observed burden separate under pandemic shock", pad=7)
    ax.grid(True)
    cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.012, shrink=0.88)
    cbar.set_label("Conversion efficiency z-score")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
        title="Label support",
        handletextpad=0.6,
        columnspacing=1.1,
        borderpad=0.45,
    )
    ax.text(0.01, 0.02, "n=106 OWID validation countries", transform=ax.transAxes, color="#444444", fontsize=7.2)
    save_figure(fig, "fig2_potential_capacity_vs_shock_burden")


def fig3_profiles() -> None:
    profiles = read_csv("results/tables/conversion_profiles.csv")
    labels = read_csv("results/tables/stage4_country_label_stability.csv")
    df = profiles.merge(labels[["iso3", "country_label_category"]], on="iso3", how="left")
    df["is_stable"] = df["country_label_category"].fillna("").str.startswith("stable")
    x_med = df["potential_capacity_score"].median()
    y_med = df["conversion_efficiency_zscore"].median()

    fig, ax = plt.subplots(figsize=(7.15, 4.95), constrained_layout=True)
    for profile, sub in df.groupby("conversion_profile", sort=False):
        ax.scatter(
            sub["potential_capacity_score"],
            sub["conversion_efficiency_zscore"],
            s=np.where(sub["is_stable"], 48, 30),
            color=PROFILE_COLORS.get(profile, "#777777"),
            edgecolor=np.where(sub["is_stable"], "#111111", "white"),
            linewidth=np.where(sub["is_stable"], 0.78, 0.28),
            alpha=0.88,
            label=f"{profile} (n={len(sub)})",
        )
    ax.axvline(x_med, color="#333333", lw=0.72, ls="--")
    ax.axhline(y_med, color="#333333", lw=0.72, ls="--")
    ax.text(x_med + 0.010, y_med + 2.05, "Effective\nconverters", color="#386cb0", weight="bold", fontsize=7.6)
    ax.text(x_med + 0.010, y_med - 2.55, "Capacity\nunder-realizers", color="#d95f02", weight="bold", fontsize=7.6)
    ax.text(x_med - 0.205, y_med + 2.05, "Adaptive\nover-performers", color="#1b9e77", weight="bold", fontsize=7.6)
    ax.text(x_med - 0.205, y_med - 2.55, "Structurally\nvulnerable", color="#7570b3", weight="bold", fontsize=7.6)
    ax.set_xlabel("Potential capacity score")
    ax.set_ylabel("Conversion efficiency z-score")
    ax.set_title("Profiles combine capacity position with realized conversion", pad=7)
    ax.grid(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=True,
        ncol=2,
        handletextpad=0.6,
        columnspacing=1.4,
        borderpad=0.45,
    )
    save_figure(fig, "fig3_conversion_efficiency_profiles")


def fig4_uncertainty_label_triage() -> None:
    stable = read_csv("results/tables/stage4_stable_case_audit.csv")
    labels = read_csv("results/tables/stage4_country_label_stability.csv")
    bottleneck = read_csv("results/tables/stage4_bottleneck_cross_evidence_stability.csv")

    selected_iso = ["BLR", "SVK", "USA", "DOM", "SUR", "DNK", "JPN", "FIN", "MEX", "PER"]
    panel_a = stable[stable["iso3"].isin(selected_iso)].copy()
    panel_a["order"] = panel_a["iso3"].map({iso: i for i, iso in enumerate(selected_iso)})
    panel_a = panel_a.sort_values("order")

    fig = plt.figure(figsize=(7.15, 6.10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.32, 1.0], hspace=0.38, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(panel_a))
    colors = [PROFILE_COLORS.get(p, "#777777") for p in panel_a["profile_label"]]
    x = panel_a["conversion_efficiency_zscore"].to_numpy()
    low = panel_a["conversion_efficiency_ci_low"].to_numpy()
    high = panel_a["conversion_efficiency_ci_high"].to_numpy()
    ax_a.hlines(y, low, high, color=colors, lw=1.65, alpha=0.95)
    ax_a.scatter(x, y, color=colors, edgecolor="#111111", s=26, zorder=3)
    ax_a.axvline(0, color="#333333", lw=0.72, ls="--")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{r.iso3}  {r.country}" for r in panel_a.itertuples()])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Conversion efficiency z-score with uncertainty interval")
    ax_a.set_title("A. Selected stable-case conversion intervals", pad=5)
    ax_a.grid(True, axis="x")

    counts = {
        "Stable": int(labels["country_label_category"].fillna("").str.startswith("stable").sum()),
        "Evidence-\ndependent": int((labels["country_label_category"] == "evidence-dependent label").sum()),
        "Data-\nlimited": int((labels["country_label_category"] == "data-limited").sum()),
    }
    ax_b.bar(counts.keys(), counts.values(), color=["#386cb0", "#d95f02", "#8c8c8c"])
    ax_b.set_ylabel("Countries")
    ax_b.set_title("B. Label triage", pad=5)
    for idx, val in enumerate(counts.values()):
        ax_b.text(idx, val + 2, str(val), ha="center", va="bottom", fontsize=7.4)
    ax_b.set_ylim(0, max(counts.values()) * 1.22)
    ax_b.grid(True, axis="y")

    comparison_labels = [
        "OWID vs\nGHS",
        "OWID vs\nlife exp.",
        "GHS vs\nlife exp.",
    ]
    ax_c.bar(
        comparison_labels,
        bottleneck["bottleneck_block_agreement"],
        color=["#4c78a8", "#f58518", "#54a24b"],
    )
    ax_c.set_ylim(0, 1.14)
    ax_c.set_ylabel("Bottleneck agreement")
    ax_c.set_title("C. Cross-evidence bottleneck agreement", pad=5)
    for idx, val in enumerate(bottleneck["bottleneck_block_agreement"]):
        ax_c.text(idx, min(val + 0.035, 1.10), f"{val:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax_c.grid(True, axis="y")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.09)

    save_figure(fig, "fig4_uncertainty_label_triage")


def fig5_cross_evidence() -> None:
    ce = read_csv("results/tables/stage4_cross_evidence_profile_stability.csv")
    under = read_csv("results/tables/stage4_under_realizer_cross_evidence_overlap.csv")
    over = read_csv("results/tables/stage4_over_performer_cross_evidence_overlap.csv")

    variants = ["baseline_owid_excess_mortality", "ghs_enhanced_owid", "life_expectancy_loss"]
    labels = ["OWID", "GHS-enh.", "Life exp."]
    idx = {v: i for i, v in enumerate(variants)}
    mat = np.eye(3)
    spearman = np.eye(3)
    for r in ce.itertuples():
        i, j = idx[r.evidence_variant_a], idx[r.evidence_variant_b]
        mat[i, j] = mat[j, i] = r.profile_agreement
        spearman[i, j] = spearman[j, i] = r.conversion_efficiency_spearman

    fig = plt.figure(figsize=(7.15, 5.40))
    ax_a = fig.add_axes([0.08, 0.58, 0.30, 0.31])
    cax = fig.add_axes([0.405, 0.58, 0.016, 0.31])
    ax_b = fig.add_axes([0.55, 0.58, 0.39, 0.31])
    ax_c = fig.add_axes([0.08, 0.12, 0.86, 0.31])

    im = ax_a.imshow(mat, cmap="YlGnBu", vmin=0.55, vmax=1.0)
    ax_a.set_xticks(range(3), labels, rotation=35, ha="right")
    ax_a.set_yticks(range(3), labels)
    ax_a.set_title("A. Profile agreement matrix", pad=5)
    for i in range(3):
        for j in range(3):
            text_color = "white" if mat[i, j] >= 0.90 else "#111111"
            ax_a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=7.6)
    fig.colorbar(im, cax=cax)

    pair_labels = ["OWID-GHS", "OWID-life", "GHS-life"]
    x = np.arange(len(pair_labels))
    width = 0.36
    ax_b.bar(x - width / 2, under["top10_under_realizer_overlap"], width, label="Under-realizer", color="#d95f02")
    ax_b.bar(x + width / 2, over["top10_over_performer_overlap"], width, label="Over-performer", color="#1b9e77")
    ax_b.set_xticks(x, pair_labels, rotation=25, ha="right")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel("Top-10 overlap", labelpad=8)
    ax_b.set_title("B. Extreme-case overlap", pad=5)
    ax_b.legend(frameon=True, borderpad=0.35, handlelength=1.6)
    ax_b.grid(True, axis="y")

    x2 = np.arange(len(pair_labels))
    ax_c.bar(x2 - width / 2, ce["profile_agreement"], width, label="Profile agreement", color="#4c78a8")
    ax_c.bar(x2 + width / 2, ce["conversion_efficiency_spearman"], width, label="Spearman", color="#f58518")
    ax_c.set_xticks(x2, pair_labels)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("Agreement")
    ax_c.set_title("C. Cross-evidence agreement summary", pad=5)
    ax_c.legend(frameon=True, ncol=2, borderpad=0.35, handlelength=1.6)
    ax_c.grid(True, axis="y")

    save_figure(fig, "fig5_cross_evidence_stability")


def supp_figures() -> None:
    bottleneck = read_csv("results/tables/conversion_bottleneck_scores.csv")
    pivot = pd.crosstab(
        bottleneck["conversion_profile"],
        bottleneck["weakest_bottleneck_block"],
        normalize="index",
    ).reindex(list(PROFILE_COLORS)).fillna(0)
    fig, ax = plt.subplots(figsize=(7.15, 4.35))
    im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=max(0.45, pivot.values.max()))
    ax.set_xticks(range(len(pivot.columns)), [c.replace("_", " ") for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), [wrapped_label(i, 24) for i in pivot.index])
    ax.set_title("Supplementary bottleneck heatmap by profile", pad=6)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:.2f}", ha="center", va="center", fontsize=7.0)
    fig.colorbar(im, ax=ax, label="Share of profile")
    save_figure(fig, "supp_fig1_bottleneck_heatmap")

    bench = read_csv("results/tables/benchmark_comparison.csv")
    keep = [
        "UHC service coverage index",
        "Equal-weight block composite",
        "PCA block composite",
        "Ridge regression predictor",
        "Random forest predictor",
        "single CRITIC-MARCOS pipeline",
        "entropy-TOPSIS pipeline",
    ]
    b = bench[bench["model_or_benchmark"].isin(keep)].copy()
    b["short"] = b["model_or_benchmark"].map(
        {
            "UHC service coverage index": "UHC",
            "Equal-weight block composite": "Equal-weight",
            "PCA block composite": "PCA",
            "Ridge regression predictor": "Ridge",
            "Random forest predictor": "RF audit",
            "single CRITIC-MARCOS pipeline": "CRITIC-MARCOS",
            "entropy-TOPSIS pipeline": "Entropy-TOPSIS",
        }
    )
    fig, ax = plt.subplots(figsize=(7.15, 4.25))
    colors = ["#999999" if x != "RF audit" else "#d95f02" for x in b["short"]]
    ax.bar(b["short"], b["spearman_with_shock_burden"], color=colors)
    ax.set_ylabel("Spearman with observed burden")
    ax.set_title("Supplementary benchmark comparison; RF shown as sensitivity audit")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "supp_fig2_benchmark_rf_sensitivity")

    rf = read_csv("results/tables/rf_benchmark_audit.csv")
    imp = read_csv("results/tables/rf_permutation_importance.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 3.65), gridspec_kw={"width_ratios": [1.05, 1.0]})
    rf_plot = rf[rf["audit_item"].isin(["random_forest_repeated_kfold_cv", "ridge_repeated_kfold_cv", "linear_repeated_kfold_cv", "random_forest_shuffled_outcome_negative_control"])]
    ax1.barh(
        [wrapped_label(v.replace("_", " "), 22) for v in rf_plot["audit_item"]],
        rf_plot["rmse_mean"],
        xerr=rf_plot["rmse_std"].fillna(0),
        color=["#d95f02", "#4c78a8", "#999999", "#666666"],
    )
    ax1.set_xlabel("Repeated-CV RMSE")
    ax1.set_title("A. RF audit and controls")
    imp = imp.sort_values("importance_mean")
    ax2.barh(
        [v.replace("_score", "").replace("_", " ") for v in imp["feature"]],
        imp["importance_mean"],
        xerr=imp["importance_std"],
        color="#4c78a8",
    )
    ax2.set_xlabel("Permutation importance")
    ax2.set_title("B. Leakage-screened feature audit")
    save_figure(fig, "supp_fig3_rf_audit")

    prob = read_csv("results/tables/profile_probability_table.csv")
    prob_piv = prob.pivot(index=["country", "iso3"], columns="profile", values="profile_probability")
    order = (
        prob_piv.assign(maxprob=prob_piv.max(axis=1), maxprofile=prob_piv.idxmax(axis=1))
        .sort_values(["maxprofile", "maxprob"], ascending=[True, False])
        .index
    )
    prob_piv = prob_piv.loc[order, list(PROFILE_COLORS)]
    fig, ax = plt.subplots(figsize=(7.15, 7.25))
    im = ax.imshow(prob_piv.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(prob_piv.columns)), [wrapped_label(c, 18) for c in prob_piv.columns], rotation=35, ha="right")
    tick_idx = list(range(0, len(prob_piv.index), 3))
    ax.set_yticks(tick_idx, [f"{prob_piv.index[i][1]}" for i in tick_idx])
    ax.set_title("Supplementary full profile probability heatmap")
    fig.colorbar(im, ax=ax, label="Profile probability")
    save_figure(fig, "supp_fig4_profile_probability_heatmap")

    supp_fig5_missingness()


def supp_fig5_missingness() -> None:
    block = read_csv("results/tables/block_missingness.csv")
    md = read_csv("data/processed/indicator_metadata_expanded.csv")
    md = md[md["use_as_feature_in_expanded_score"] == True].copy()
    top_missing = md.sort_values("missing_count", ascending=False).head(10)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.15, 6.10),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 2.15]},
    )
    ax1.bar(block["block"].str.replace("_", " "), block["mean_country_missing_rate"], color="#4c78a8")
    ax1.set_ylabel("Mean country missing rate")
    ax1.set_title("A. Block missingness", pad=5)
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.25)

    labels = [wrapped_label(v, 36) for v in top_missing["label"]]
    y_pos = np.arange(len(top_missing))
    ax2.barh(y_pos, top_missing["missing_count"], color="#f58518")
    ax2.set_yticks(y_pos, labels)
    ax2.invert_yaxis()
    ax2.axvline(0, color="#333333", linewidth=0.8)
    ax2.set_xlabel("Raw missing count")
    ax2.set_title("B. Indicators with highest missingness", pad=5)
    ax2.tick_params(axis="y", labelsize=6.8, pad=2)
    ax2.grid(True, axis="x", alpha=0.25)
    save_figure(fig, "supp_fig5_feature_missingness_block_coverage")


def write_table(path: str, content: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / path).write_text(content.strip() + "\n")


def table1_feature_blocks() -> None:
    md = read_csv("data/processed/indicator_metadata_expanded.csv")
    md = md[md["use_as_feature_in_expanded_score"] == True].copy()
    examples = {
        "capacity": "physicians, nurses, hospital beds, health expenditure",
        "equity_access": "UHC service coverage, out-of-pocket spending, sanitation, water",
        "preparedness": "measles and DPT immunization, TB incidence, HIV prevalence",
        "vulnerability": "age structure, density, NCD mortality, pollution, baseline mortality",
    }
    interpretations = {
        "capacity": "Resource and service-capacity evidence available before the shock.",
        "equity_access": "Coverage, access, and financial-protection conditions that shape realized use.",
        "preparedness": "Pre-shock infectious-disease control and routine-readiness proxies.",
        "vulnerability": "Demographic, environmental, and baseline-health context used to benchmark burden.",
    }
    order = ["capacity", "equity_access", "preparedness", "vulnerability"]
    rows = []
    for block in order:
        sub = md[md["block"] == block]
        rows.append(
            f"{latex_escape({'capacity': 'Capacity', 'equity_access': 'Equity and access', 'preparedness': 'Preparedness', 'vulnerability': 'Vulnerability'}[block])} & {len(sub)} & 193 & "
            f"{latex_escape(examples[block])} & {latex_escape(interpretations[block])} \\\\"
        )
    content = r"""
\begin{table}[!htbp]
\centering
\caption{Pre-shock feature blocks used by HSR-Convert for the 193-country scoring frame. Indicators are observed before 2020 and grouped by their decision-support role.}
\label{tab:feature-blocks}
\small
\begin{tabularx}{\linewidth}{lrrYY}
\toprule
Block & Indicators & Countries & Example indicators & Interpretation \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    write_table("table1_feature_blocks_summary.tex", content)


def table2_models() -> None:
    df = read_csv("paper_assets/tables/table2_conversion_model_diagnostics.csv")
    df = df[df["model_version"].isin(["linear", "ridge", "huber"])].copy()
    role = {
        "linear": "Transparent baseline for expected-burden modeling.",
        "ridge": "Primary model; shrinkage improves stability in a modest country sample.",
        "huber": "Robust linear sensitivity for residual conversion scores.",
    }
    rows = []
    for r in df.itertuples():
        rows.append(
            f"{latex_escape(r.model_version.title())} & {int(r.n)} & {r.cv_rmse:.1f} & "
            f"{r.r2:.3f} & {r.adjusted_r2:.3f} & {latex_escape(role[r.model_version])} \\\\"
        )
    content = r"""
\begin{table}[!htbp]
\centering
\caption{Expected-burden model diagnostics for OWID validation countries. Ridge regression is retained as the primary model; nonlinear random-forest results are reported only in the supplement as a sensitivity audit.}
\label{tab:model-diagnostics}
\small
\begin{tabularx}{\linewidth}{lrrrrY}
\toprule
Model & Countries & CV RMSE & $R^2$ & Adjusted $R^2$ & Role in HSR-Convert \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    write_table("table2_model_diagnostics_summary.tex", content)


def table3_profiles() -> None:
    df = read_csv("paper_assets/tables/table3_conversion_profiles_and_examples.csv")
    interpretations = {
        "adaptive over-performers": "Weaker pre-shock capacity paired with favorable residual conversion.",
        "capacity under-realizers": "Stronger pre-shock capacity paired with unfavorable residual conversion.",
        "effective converters": "Stronger pre-shock capacity and favorable residual conversion.",
        "structurally vulnerable systems": "Weaker pre-shock capacity and unfavorable residual conversion.",
        "uncertain / data-limited systems": "Profile assignment requires caution because uncertainty or evidence limits dominate.",
    }
    rows = []
    for r in df.itertuples():
        profile_label = r.conversion_profile[:1].upper() + r.conversion_profile[1:]
        rows.append(
            f"{latex_escape(profile_label)} & {int(r.n)} & {r.median_efficiency:.3f} & "
            f"{latex_escape(r.example_countries)} & {latex_escape(interpretations[r.conversion_profile])} \\\\"
        )
    content = r"""
\begin{table}[!htbp]
\centering
\caption{Conversion profiles in the OWID validation set (n=106). Profiles combine pre-shock capacity position with residual conversion efficiency rather than imposing a single country ordering.}
\label{tab:profiles}
\small
\begin{tabularx}{\linewidth}{p{0.22\linewidth}rrYY}
\toprule
Profile & Countries & Median efficiency & Example countries & Evidence-safe interpretation \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    write_table("table3_profile_summary.tex", content)


def table6_contribution_map() -> None:
    rows = [
        ("Capacity-to-resilience conversion", "Separates pre-shock capacity from shock-period realized performance.", "Figures 1--3; Tables 1--3"),
        ("Leakage-safe temporal design", "Uses 2015--2019 indicators for capacity and 2020--2022 outcomes for validation.", "Data section; Table 1"),
        ("Decision-support profiles", "Turns residual conversion into interpretable diagnostic categories.", "Figure 3; Table 3"),
        ("Uncertainty-aware label triage", "Reports stable, evidence-dependent, and data-limited labels instead of forcing universal conclusions.", "Figure 4; Table 5"),
        ("Cross-evidence stability", "Compares OWID baseline, GHS-enhanced OWID, and life-expectancy-loss lenses.", "Figure 5; Table 4"),
    ]
    content_rows = [
        f"{latex_escape(a)} & {latex_escape(b)} & {latex_escape(c)} \\\\" for a, b, c in rows
    ]
    content = r"""
\begin{table}[!htbp]
\centering
\caption{Contribution-to-evidence map for HSR-Convert as an expert decision-support architecture. The table identifies how each contribution is supported within the manuscript rather than adding new empirical claims.}
\label{tab:contribution-map}
\small
\begin{tabularx}{\linewidth}{p{0.25\linewidth}YY}
\toprule
Contribution & Decision-support function & Where supported \\
\midrule
""" + "\n".join(content_rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    write_table("table6_contribution_map.tex", content)


def supplement_tables() -> None:
    md = read_csv("data/processed/indicator_metadata_expanded.csv")
    md = md[md["use_as_feature_in_expanded_score"] == True].copy()
    order = ["capacity", "equity_access", "preparedness", "vulnerability"]
    md["block_order"] = md["block"].map({b: i for i, b in enumerate(order)})
    md = md.sort_values(["block_order", "variable"])
    rows = []
    for r in md.itertuples():
        rows.append(
            f"{latex_escape(r.block.replace('_', ' '))} & {latex_escape(r.label)} & "
            f"{latex_escape(r.source_code)} & {latex_escape(r.direction)} \\\\"
        )
    write_table(
        "supp_table_s1_indicator_list.tex",
        r"""
\begingroup
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{p{0.13\linewidth}p{0.42\linewidth}p{0.18\linewidth}p{0.13\linewidth}}
\caption{Full pre-shock indicator list used in the expanded HSR-Convert score. All listed indicators are used as scoring features and are restricted to 2015--2019 observations.}
\label{tab:supp-indicators}\\
\toprule
Block & Indicator & Source code & Orientation \\
\midrule
\endfirsthead
\toprule
Block & Indicator & Source code & Orientation \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}
\endgroup
""",
    )

    block = read_csv("results/tables/block_missingness.csv")
    rows = []
    for r in block.itertuples():
        rows.append(
            f"{latex_escape(r.block.replace('_', ' '))} & {int(r.n_indicators)} & {int(r.n_countries)} & "
            f"{r.mean_country_missing_rate:.3f} & {r.median_country_missing_rate:.3f} & "
            f"{int(r.countries_with_complete_block)} & {int(r.countries_missing_entire_block)} \\\\"
        )
    write_table(
        "supp_table_s2_block_missingness.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Missingness by pre-shock feature block before imputation. The audit is used to interpret data-limited labels and block reliability.}
\label{tab:supp-missingness}
\footnotesize
\begin{tabularx}{\linewidth}{p{0.18\linewidth}rrrrrr}
\toprule
Block & Indicators & Countries & Mean & Median & Complete & Entire missing \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    ghs = read_csv("results/tables/stage4_ghs_2019_integration_audit.csv").iloc[0]
    rows = [
        ("Status", ghs["status"]),
        ("Matched ISO3 countries", int(ghs["n_matched_iso3_countries"])),
        ("Missing scoring countries", int(ghs["missing_count"])),
        ("Missing rate", f"{ghs['missing_rate']:.3f}"),
        ("SHA-256", f"{str(ghs['file_hash_sha256'])[:16]}...{str(ghs['file_hash_sha256'])[-16:]}"),
        ("Source", "official GHS raw CSV from the GHS Index website"),
    ]
    write_table(
        "supp_table_s3_ghs_audit.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{GHS 2019 integration audit. The file is used as pre-shock preparedness evidence; GHS 2021 is not used as a pre-shock predictor.}
\label{tab:supp-ghs}
\footnotesize
\begin{tabularx}{\linewidth}{lY}
\toprule
Audit item & Value \\
\midrule
""" + "\n".join(f"{latex_escape(a)} & {latex_escape(b)} \\\\" for a, b in rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    le_cov = read_csv("results/tables/stage4_life_expectancy_coverage.csv")
    le_val = read_csv("results/tables/stage4_life_expectancy_secondary_validation.csv")
    merged = le_cov.merge(le_val[["outcome", "conversion_efficiency_correlation_with_owid", "under_realizer_top10_overlap_with_owid", "over_performer_top10_overlap_with_owid"]], on="outcome", how="left")
    rows = []
    outcome_labels = {
        "life_expectancy_loss_2019_2020": "2019--2020 loss",
        "life_expectancy_loss_2019_2021": "2019--2021 loss",
        "life_expectancy_loss_2019_2022": "2019--2022 loss",
        "max_life_expectancy_drop_2020_2022": "Max 2020--2022 drop",
    }
    for r in merged.itertuples():
        rows.append(
            f"{latex_escape(outcome_labels.get(r.outcome, r.outcome.replace('_', ' ')))} & {int(r.coverage_count)} & {r.missing_rate:.3f} & "
            f"{r.conversion_efficiency_correlation_with_owid:.3f} & {r.under_realizer_top10_overlap_with_owid:.2f} & {r.over_performer_top10_overlap_with_owid:.2f} \\\\"
        )
    write_table(
        "supp_table_s4_life_expectancy.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Life-expectancy-loss construction and secondary validation. All variants are derived from World Bank SP.DYN.LE00.IN downloads for 2019--2022.}
\label{tab:supp-life-expectancy}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{p{0.28\linewidth}rrrrr}
\toprule
Outcome & Coverage & Missing & Spearman & Under & Over \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    rf = read_csv("results/tables/rf_benchmark_audit.csv")
    audit_labels = {
        "random_forest_repeated_kfold_cv": "RF repeated CV",
        "ridge_repeated_kfold_cv": "Ridge repeated CV",
        "linear_repeated_kfold_cv": "Linear repeated CV",
        "random_forest_shuffled_outcome_negative_control": "RF shuffled-outcome control",
    }
    status_labels = {
        "sensitivity_only": "sensitivity",
        "negative_control": "negative control",
    }
    rows = []
    for r in rf.itertuples():
        rows.append(
            f"{latex_escape(audit_labels.get(r.audit_item, r.audit_item.replace('_', ' ')))} & {latex_escape(r.model.replace('_', ' '))} & {int(r.n)} & "
            f"{str(bool(r.no_leakage_detected))} & {r.rmse_mean:.1f} & {r.r2_cv:.3f} & {latex_escape(status_labels.get(r.status, r.status))} \\\\"
        )
    write_table(
        "supp_table_s5_rf_audit.tex",
        r"""
\begin{table}[!htbp]
\centering
\caption{Random-forest sensitivity audit. RF is included to test nonlinear sensitivity and leakage screens, not as the primary HSR-Convert model.}
\label{tab:supp-rf}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\linewidth}{p{0.24\linewidth}p{0.16\linewidth}rrrrY}
\toprule
Audit item & Model & Countries & No leakage & RMSE & CV $R^2$ & Status \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
""",
    )


def main() -> None:
    setup_style()
    fig1_architecture()
    fig2_capacity_vs_burden()
    fig3_profiles()
    fig4_uncertainty_label_triage()
    fig5_cross_evidence()
    supp_figures()
    table1_feature_blocks()
    table2_models()
    table3_profiles()
    table6_contribution_map()
    supplement_tables()


if __name__ == "__main__":
    main()
