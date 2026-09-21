"""A1: sensitivity of the heuristic design choices — block/indicator weights, profile thresholds, Monte Carlo settings [R2-4]."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision_analysis.common import (
    OUTCOME,
    ROOT,
    RESULTS_DIR,
    SEED,
    baseline_profiles,
    efficiency_from_predictions,
    ensure_results_dirs,
    make_model,
    update_numbers,
)
from src.conversion_monte_carlo import run_conversion_monte_carlo
from src.conversion_robustness import encoded_xy, load_conversion_model_data, model_feature_columns
from src.imputation import impute_matrix
from src.normalization import normalize_matrix

BLOCKS = ["capacity", "preparedness", "vulnerability", "equity_access"]
N_WEIGHT_DRAWS = 500
PERCENTILE_GRID = [40, 45, 50, 55, 60]


# ---------------------------------------------------------------------------------------------
# A1.1 block-weight and within-block-weight perturbation
# ---------------------------------------------------------------------------------------------
def _normalized_blocks(root=ROOT) -> dict[str, pd.DataFrame]:
    """Per-block oriented, median-imputed, robust-min-max indicator matrices (as in feature_blocks._block_point_scores)."""

    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    meta = pd.read_csv(root / "data/processed/indicator_metadata_expanded.csv")
    feature_meta = meta[meta["use_as_feature_in_expanded_score"].astype(bool)]
    sample = matrix[matrix["expanded_primary_sample"].astype(bool)].copy()
    out = {}
    for block, block_meta in feature_meta.groupby("block"):
        cols = block_meta["variable"].tolist()
        directions = block_meta.set_index("variable")["direction"].to_dict()
        raw = sample.set_index("iso3")[cols]
        out[block] = normalize_matrix(impute_matrix(raw, "median"), directions, "robust_minmax")
    return out


def _capacity_from_weights(blocks: dict[str, pd.DataFrame], alpha: np.ndarray, within: dict[str, np.ndarray]) -> pd.DataFrame:
    scores = pd.DataFrame(index=blocks[BLOCKS[0]].index)
    for b, name in enumerate(BLOCKS):
        scores[f"{name}_score"] = blocks[name].to_numpy() @ within[name]
    scores["potential_capacity_score"] = scores[[f"{n}_score" for n in BLOCKS]].to_numpy() @ alpha
    return scores.reset_index()


def _conversion_with_capacity(data: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    sim = data.drop(columns=["potential_capacity_score", "vulnerability_score"]).merge(
        scores[["iso3", "potential_capacity_score", "vulnerability_score"]], on="iso3", how="left"
    )
    x, y, work = encoded_xy(sim, outcome_col=OUTCOME, feature_cols=model_feature_columns())
    work = work.rename(columns={OUTCOME: "observed_shock_burden"}).reset_index(drop=True)
    model = make_model("ridge", x).fit(x, y)
    return efficiency_from_predictions(work, model.predict(x))


def weight_sensitivity(root=ROOT) -> pd.DataFrame:
    blocks = _normalized_blocks(root)
    data = load_conversion_model_data(root)
    base = baseline_profiles(root).set_index("iso3")
    base_scores = _capacity_from_weights(
        blocks, np.full(len(BLOCKS), 0.25), {n: np.full(blocks[n].shape[1], 1.0 / blocks[n].shape[1]) for n in BLOCKS}
    )
    check = _conversion_with_capacity(data, base_scores).set_index("iso3")
    assert (check["conversion_profile"] == base.loc[check.index, "conversion_profile"]).all(), "equal-weight reconstruction must reproduce the baseline"
    rng = np.random.default_rng(SEED)
    rows = []
    label_rows = []
    schemes = [
        ("dirichlet_moderate", 20.0, 20.0),  # concentration 20 x default weights: sd of a block weight about 0.09
        ("dirichlet_flat", 1.0, 1.0),  # flat Dirichlet(1,..,1): the package Monte Carlo setting, much wider
        ("block_only_flat", 1.0, None),  # block weights flat, within-block weights kept equal
    ]
    for scheme, conc_block, conc_within in schemes:
        for d in range(N_WEIGHT_DRAWS):
            alpha = rng.dirichlet(np.full(len(BLOCKS), conc_block * 0.25 if scheme != "dirichlet_flat" else 1.0))
            within = {}
            for name in BLOCKS:
                k = blocks[name].shape[1]
                if conc_within is None:
                    within[name] = np.full(k, 1.0 / k)
                elif scheme == "dirichlet_flat":
                    within[name] = rng.dirichlet(np.ones(k))
                else:
                    within[name] = rng.dirichlet(np.full(k, conc_within / k))
            scores = _capacity_from_weights(blocks, alpha, within)
            conv = _conversion_with_capacity(data, scores).set_index("iso3")
            common = conv.index.intersection(base.index)
            same = conv.loc[common, "conversion_profile"] == base.loc[common, "conversion_profile"]
            rows.append(
                {
                    "scheme": scheme,
                    "draw": d,
                    "alpha_capacity": alpha[0],
                    "alpha_preparedness": alpha[1],
                    "alpha_vulnerability": alpha[2],
                    "alpha_equity_access": alpha[3],
                    "profile_agreement_with_baseline": float(same.mean()),
                    "spearman_efficiency_vs_baseline": float(spearmanr(conv.loc[common, "conversion_efficiency_zscore"], base.loc[common, "conversion_efficiency_zscore"]).statistic),
                    "spearman_capacity_vs_baseline": float(spearmanr(conv.loc[common, "potential_capacity_score"], base.loc[common, "potential_capacity_score"]).statistic),
                }
            )
            for iso3 in common:
                label_rows.append({"scheme": scheme, "draw": d, "iso3": iso3, "profile": conv.loc[iso3, "conversion_profile"], "same_as_baseline": bool(same.loc[iso3])})
    draws = pd.DataFrame(rows)
    labels = pd.DataFrame(label_rows)
    per_country = (
        labels.groupby(["scheme", "iso3"])
        .agg(flip_rate=("same_as_baseline", lambda s: 1.0 - s.mean()), modal_profile=("profile", lambda s: s.value_counts().index[0]))
        .reset_index()
    )
    per_country = per_country.merge(base[["country", "conversion_profile"]].reset_index(), on="iso3", how="left")
    per_country["modal_equals_baseline"] = per_country["modal_profile"] == per_country["conversion_profile"]
    draws.to_csv(RESULTS_DIR / "sensitivity_weights_draws.csv", index=False)
    per_country.to_csv(RESULTS_DIR / "sensitivity_weights.csv", index=False)
    summary = {}
    for scheme, sub in draws.groupby("scheme"):
        pc = per_country[per_country.scheme.eq(scheme)]
        summary[scheme] = {
            "n_draws": int(len(sub)),
            "mean_profile_agreement": float(sub["profile_agreement_with_baseline"].mean()),
            "p05_profile_agreement": float(sub["profile_agreement_with_baseline"].quantile(0.05)),
            "min_profile_agreement": float(sub["profile_agreement_with_baseline"].min()),
            "mean_spearman_efficiency": float(sub["spearman_efficiency_vs_baseline"].mean()),
            "p05_spearman_efficiency": float(sub["spearman_efficiency_vs_baseline"].quantile(0.05)),
            "mean_spearman_capacity": float(sub["spearman_capacity_vs_baseline"].mean()),
            "share_countries_never_flip": float(pc["flip_rate"].eq(0).mean()),
            "share_countries_flip_le_10pct": float(pc["flip_rate"].le(0.10).mean()),
            "share_countries_modal_equals_baseline": float(pc["modal_equals_baseline"].mean()),
            "median_country_flip_rate": float(pc["flip_rate"].median()),
            "n_countries": int(len(pc)),
        }
    return draws, per_country, summary


# ---------------------------------------------------------------------------------------------
# A1.2 profile-threshold grid
# ---------------------------------------------------------------------------------------------
def profiles_with_cuts(df: pd.DataFrame, cap_pct: float, eff_pct: float | None, burden_pct: float) -> pd.Series:
    """Package profile rule with movable cut-points. eff_pct=None keeps the baseline E>=0 rule."""

    cap_cut = np.percentile(df["potential_capacity_score"], cap_pct)
    burden_cut = np.percentile(df["observed_shock_burden"], burden_pct)
    eff_cut = 0.0 if eff_pct is None else np.percentile(df["conversion_efficiency_zscore"], eff_pct)
    disagreement_cut = df["method_disagreement_index"].quantile(0.75)
    missing_cut = max(0.20, df["expanded_feature_missing_rate"].quantile(0.75))

    def label(row: pd.Series) -> str:
        if row["method_disagreement_index"] >= disagreement_cut or row["expanded_feature_missing_rate"] > missing_cut:
            return "uncertain / data-limited systems"
        high_capacity = row["potential_capacity_score"] >= cap_cut
        high_burden = row["observed_shock_burden"] > burden_cut
        positive_eff = row["conversion_efficiency_zscore"] >= eff_cut
        if high_capacity and not high_burden and positive_eff:
            return "effective converters"
        if high_capacity and (high_burden or not positive_eff):
            return "capacity under-realizers"
        if (not high_capacity) and positive_eff:
            return "adaptive over-performers"
        return "structurally vulnerable systems"

    return df.apply(label, axis=1)


def threshold_sensitivity(root=ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = baseline_profiles(root).copy()
    check = profiles_with_cuts(base, 50, None, 50)
    assert (check == base["conversion_profile"]).all(), "median cuts with E>=0 must reproduce the baseline"
    base["baseline_eff_percentile"] = float((base["conversion_efficiency_zscore"] < 0).mean() * 100)
    rows = []
    for cap_pct, eff_pct, burden_pct in product(PERCENTILE_GRID, PERCENTILE_GRID, PERCENTILE_GRID):
        labels = profiles_with_cuts(base, cap_pct, eff_pct, burden_pct)
        for iso3, lab, b in zip(base["iso3"], labels, base["conversion_profile"]):
            rows.append({"cap_pct": cap_pct, "eff_pct": eff_pct, "burden_pct": burden_pct, "iso3": iso3, "profile": lab, "same_as_baseline": lab == b})
    grid = pd.DataFrame(rows)
    grid["in_main_25_grid"] = grid["burden_pct"].eq(50)
    per_combo = grid.groupby(["cap_pct", "eff_pct", "burden_pct"])["same_as_baseline"].mean().reset_index(name="share_countries_same_as_baseline")
    per_country_25 = grid[grid.in_main_25_grid].groupby("iso3").agg(stability_25=("same_as_baseline", "mean"), n_distinct_labels_25=("profile", "nunique")).reset_index()
    per_country_125 = grid.groupby("iso3").agg(stability_125=("same_as_baseline", "mean"), n_distinct_labels_125=("profile", "nunique")).reset_index()
    per_country = per_country_25.merge(per_country_125, on="iso3").merge(base[["iso3", "country", "conversion_profile"]], on="iso3")
    per_combo.to_csv(RESULTS_DIR / "sensitivity_thresholds_grid.csv", index=False)
    per_country.to_csv(RESULTS_DIR / "sensitivity_thresholds.csv", index=False)
    main25 = per_combo[per_combo.burden_pct.eq(50)]
    non_uncertain = base["conversion_profile"].ne("uncertain / data-limited systems")
    summary = {
        "grid_percentiles": PERCENTILE_GRID,
        "baseline_efficiency_cut_percentile": float(base["baseline_eff_percentile"].iloc[0]),
        "n_combinations_main_grid": int(len(main25)),
        "n_combinations_full_grid": int(len(per_combo)),
        "mean_share_same_as_baseline_main_grid": float(main25["share_countries_same_as_baseline"].mean()),
        "min_share_same_as_baseline_main_grid": float(main25["share_countries_same_as_baseline"].min()),
        "mean_share_same_as_baseline_full_grid": float(per_combo["share_countries_same_as_baseline"].mean()),
        "min_share_same_as_baseline_full_grid": float(per_combo["share_countries_same_as_baseline"].min()),
        "n_countries_invariant_main_grid": int(per_country["stability_25"].eq(1).sum()),
        "share_countries_invariant_main_grid": float(per_country["stability_25"].eq(1).mean()),
        "n_countries_invariant_full_grid": int(per_country["stability_125"].eq(1).sum()),
        "share_countries_invariant_full_grid": float(per_country["stability_125"].eq(1).mean()),
        "n_non_uncertain_countries": int(non_uncertain.sum()),
        "n_non_uncertain_invariant_main_grid": int(per_country.loc[non_uncertain.values, "stability_25"].eq(1).sum()),
        "share_non_uncertain_invariant_main_grid": float(per_country.loc[non_uncertain.values, "stability_25"].eq(1).mean()),
        "n_countries_invariant_at_most_2_labels_full_grid": int(per_country["n_distinct_labels_125"].le(2).sum()),
        "n_countries": int(len(per_country)),
    }
    # stable-label subset (cross-evidence stable labels) under the grid
    stab = pd.read_csv(root / "results/tables/stage4_country_label_stability.csv")
    stable_iso = set(stab.loc[stab["country_label_category"].str.startswith("stable"), "iso3"])
    st = per_country[per_country.iso3.isin(stable_iso)]
    summary["n_stable_labels"] = int(len(st))
    summary["n_stable_labels_invariant_main_grid"] = int(st["stability_25"].eq(1).sum())
    summary["mean_stability_stable_labels_main_grid"] = float(st["stability_25"].mean())
    summary["mean_stability_stable_labels_full_grid"] = float(st["stability_125"].mean())
    return per_combo, per_country, summary


# ---------------------------------------------------------------------------------------------
# A1.3 Monte Carlo settings
# ---------------------------------------------------------------------------------------------
def mc_sensitivity(root=ROOT) -> tuple[pd.DataFrame, dict]:
    base = baseline_profiles(root).set_index("iso3")
    rows = []
    for mult in [0.5, 1.0, 2.0]:
        for n_sim in [250, 500, 1000]:
            summary, _ = run_conversion_monte_carlo(root, n_sim=n_sim, noise_multiplier=mult, seed=SEED, save=False)
            summary = summary.set_index("iso3")
            common = summary.index.intersection(base.index)
            rows.append(
                {
                    "noise_multiplier": mult,
                    "n_simulations": n_sim,
                    "robust_profile_share": float(summary["robustness_category"].eq("robust").mean()),
                    "mean_profile_stability": float(summary["profile_stability_score"].mean()),
                    "median_profile_stability": float(summary["profile_stability_score"].median()),
                    "most_likely_profile_agreement_with_baseline": float((summary.loc[common, "most_likely_profile"] == base.loc[common, "conversion_profile"]).mean()),
                    "mean_ci_width": float((summary["conversion_efficiency_ci_high"] - summary["conversion_efficiency_ci_low"]).mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sensitivity_mc.csv", index=False)
    summary = {
        "settings": "noise multiplier in {0.5, 1, 2} applied to block-score noise (sd 0.025) and method noise; draws in {250, 500, 1000}; block weights Dirichlet(1); bootstrap refit",
        "robust_share_range": [float(out["robust_profile_share"].min()), float(out["robust_profile_share"].max())],
        "mean_stability_range": [float(out["mean_profile_stability"].min()), float(out["mean_profile_stability"].max())],
        "most_likely_agreement_range": [float(out["most_likely_profile_agreement_with_baseline"].min()), float(out["most_likely_profile_agreement_with_baseline"].max())],
        "rows": out.to_dict("records"),
    }
    return out, summary


def run() -> None:
    ensure_results_dirs()
    _, _, w = weight_sensitivity()
    _, _, t = threshold_sensitivity()
    _, m = mc_sensitivity()
    update_numbers("sensitivity", {"weights": w, "thresholds": t, "monte_carlo": m, "n_weight_draws_per_scheme": N_WEIGHT_DRAWS, "seed": SEED})
    print("weights:", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in w.items()})
    print("thresholds:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
    print("mc:", m["robust_share_range"], m["mean_stability_range"], m["most_likely_agreement_range"])


if __name__ == "__main__":
    run()
