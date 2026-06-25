#src/uncertainty.py - monte carlo uncertainty + sobol sensitivity for site LCOE
#(block 2 phase 3).
#
#forward-propagation monte carlo: sample the uncertain inputs (capex components,
#discount rate, operational losses, inter-annual capacity factor), push each draw
#through the deterministic lcoe model, and report P10/P50/P90 per site. a sobol
#variance decomposition (salib) then ranks which inputs drive lcoe variance.
#
#central case matches the deterministic baseline: the discount-rate mode is 7%
#and the capex multipliers are centred on 1.0, so the median tracks the headline
#lcoe. ranges live in config.UQ.

import numpy as np
import pandas as pd


#the ten uncertain inputs, in a fixed order shared by the mc and sobol paths
_CAPEX_KEYS = ["turbine", "foundation", "array", "transmission", "installation"]
_PARAM_NAMES = (["discount"] + [f"capex_{k}" for k in _CAPEX_KEYS]
                + ["loss_array", "loss_electrical", "loss_downtime", "cf_factor"])


def _lcoe(components: dict, capex_mult: dict, discount,
          gross_cf, cf_factor, loss_array, loss_elec, loss_down,
          rated_mw: float, opex_pct: float, lifetime: int):
    """
    vectorised per-turbine lcoe for arrays of sampled inputs.

    mirrors the deterministic model: capex from perturbed components, crf from the
    sampled discount rate, opex a fixed fraction of capex, and net aep from the
    gross cf scaled by the inter-annual factor and the three operational losses
    (the single-pass loss chain, post bug #1). availability is not re-applied.
    """
    capex = sum(components[k] * capex_mult[k] for k in _CAPEX_KEYS)
    crf = discount * (1 + discount)**lifetime / ((1 + discount)**lifetime - 1)
    opex = capex * opex_pct
    loss_factor = (1 - loss_array / 100.0) * (1 - loss_elec / 100.0) * (1 - loss_down / 100.0)
    aep = rated_mw * 8760.0 * (gross_cf * cf_factor) * loss_factor
    return (capex * crf + opex) / aep


def _site_inputs(row, rated_mw: float):
    """baseline capex components and gross cf for one candidate site."""
    from src.economics import estimate_capex

    depth = row["depth_m"]
    dist_port = row.get("dist_to_port_km", 0.0)
    dist_grid = row.get("dist_to_shore_km", dist_port)
    if pd.isna(dist_grid):
        dist_grid = dist_port

    _, components = estimate_capex(depth, rated_mw,
                                   dist_to_port_km=dist_port,
                                   dist_to_grid_km=dist_grid,
                                   return_components=True)
    return components, float(row["mean_cf"])


#monte carlo sampling

def _mc_draws(n: int, rng, U: dict) -> dict:
    """iid draws of every uncertain input for the plain monte carlo percentiles."""
    return {
        "discount":   rng.triangular(*U["discount_rate_tri"], n),
        "capex_mult": {k: rng.triangular(1 - f, 1.0, 1 + f, n)
                       for k, f in U["capex_pct"].items()},
        "loss_array": np.clip(rng.normal(*U["array_loss"], n), 0.0, None),
        "loss_elec":  np.clip(rng.normal(*U["electrical_loss"], n), 0.0, None),
        "loss_down":  np.clip(rng.normal(*U["downtime_loss"], n), 0.0, None),
        "cf_factor":  rng.normal(*U["cf_interannual"], n),
    }


#sobol sampling: transform a unit saltelli design to the input distributions

def _triang_ppf(u, lo, mode, hi):
    from scipy.stats import triang
    c = (mode - lo) / (hi - lo)
    return triang.ppf(u, c, loc=lo, scale=hi - lo)


def _transform_unit(X: np.ndarray, U: dict) -> dict:
    """map a (M, 10) saltelli design in [0,1] onto the input distributions."""
    from scipy.stats import norm

    cols = {"discount": _triang_ppf(X[:, 0], *U["discount_rate_tri"])}
    for i, k in enumerate(_CAPEX_KEYS):
        f = U["capex_pct"][k]
        cols[f"capex_{k}"] = _triang_ppf(X[:, 1 + i], 1 - f, 1.0, 1 + f)
    cols["loss_array"]      = np.clip(norm.ppf(X[:, 6], *U["array_loss"]), 0.0, None)
    cols["loss_electrical"] = np.clip(norm.ppf(X[:, 7], *U["electrical_loss"]), 0.0, None)
    cols["loss_downtime"]   = np.clip(norm.ppf(X[:, 8], *U["downtime_loss"]), 0.0, None)
    cols["cf_factor"]       = norm.ppf(X[:, 9], *U["cf_interannual"])
    return cols


def run_uncertainty(sites_df: pd.DataFrame,
                    output_path: str = "outputs/charts/uncertainty_lcoe.png") -> dict:
    """
    monte carlo lcoe distributions (P10/P50/P90) per candidate site plus a sobol
    sensitivity ranking of which inputs drive lcoe variance.

    args:
        sites_df:    candidate sites with lat, lon, depth_m, mean_cf, distances,
                     and (optionally) lcoe_gbp_per_mwh for the deterministic ref.
        output_path: png for the per-site distribution + sobol figure.

    returns a dict with the per-site percentile table and the mean sobol indices.
    """
    from config import UQ, TURBINE, ECONOMICS as E

    print("="*60)
    print("  PHASE 3 - Monte Carlo Uncertainty + Sobol Sensitivity")
    print("="*60)

    rated = TURBINE["rated_power_mw"]
    opex_pct = E["opex_pct_capex_per_year"]
    lifetime = E["project_lifetime_years"]
    rng = np.random.default_rng(UQ["seed"])

    df = sites_df.reset_index(drop=True)

    #plain monte carlo for percentiles
    draws = _mc_draws(UQ["n_samples"], rng, UQ)

    #shared saltelli design for sobol (transformed to the input distributions)
    from SALib.sample.sobol import sample as saltelli_sample
    from SALib.analyze import sobol as sobol_analyze
    problem = {"num_vars": 10, "names": _PARAM_NAMES, "bounds": [[0.0, 1.0]] * 10}
    X = saltelli_sample(problem, UQ["sobol_n"], calc_second_order=False)
    sob = _transform_unit(X, UQ)
    sob_capex_mult = {k: sob[f"capex_{k}"] for k in _CAPEX_KEYS}

    rows = []
    st_acc = np.zeros(10)
    s1_acc = np.zeros(10)

    for _, site in df.iterrows():
        components, gross_cf = _site_inputs(site, rated)

        #monte carlo percentiles
        mc = _lcoe(components, draws["capex_mult"], draws["discount"],
                   gross_cf, draws["cf_factor"],
                   draws["loss_array"], draws["loss_elec"], draws["loss_down"],
                   rated, opex_pct, lifetime)
        p10, p50, p90 = np.percentile(mc, [10, 50, 90])

        rows.append({
            "lat": site["lat"], "lon": site["lon"],
            "deterministic": site.get("lcoe_gbp_per_mwh", np.nan),
            "p10": p10, "p50": p50, "p90": p90,
            "p90_minus_p10": p90 - p10,
        })

        #sobol variance decomposition for this site
        Y = _lcoe(components, sob_capex_mult, sob["discount"],
                  gross_cf, sob["cf_factor"],
                  sob["loss_array"], sob["loss_electrical"], sob["loss_downtime"],
                  rated, opex_pct, lifetime)
        #num_resamples only affects the (unused) bootstrap CIs, not the ST/S1
        #point estimates, so keep it small for speed.
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                   num_resamples=10, print_to_console=False)
        st_acc += np.asarray(Si["ST"])
        s1_acc += np.asarray(Si["S1"])

    table = pd.DataFrame(rows)
    n_sites = len(table)
    sobol_st = st_acc / n_sites
    sobol_s1 = s1_acc / n_sites
    order = np.argsort(sobol_st)[::-1]

    print(f"[UQ] {UQ['n_samples']:,} Monte Carlo draws/site; Sobol base N={UQ['sobol_n']}.")
    print("[UQ] LCOE distribution per site (GBP/MWh):")
    print(f"  {'lat':>5} {'lon':>6} {'determ':>7} {'P10':>7} {'P50':>7} {'P90':>7} {'P90-P10':>8}")
    for _, r in table.iterrows():
        print(f"  {r['lat']:>5} {r['lon']:>6} {r['deterministic']:>7.1f} "
              f"{r['p10']:>7.1f} {r['p50']:>7.1f} {r['p90']:>7.1f} {r['p90_minus_p10']:>8.1f}")

    print("[UQ] Sobol sensitivity (mean total-order ST across sites, ranked):")
    for i in order:
        print(f"  {_PARAM_NAMES[i]:>20}  ST={sobol_st[i]:.3f}  S1={sobol_s1[i]:.3f}")

    _plot_uncertainty(table, sobol_st, output_path)

    return {
        "per_site": table,
        "sobol_st": dict(zip(_PARAM_NAMES, sobol_st)),
        "sobol_s1": dict(zip(_PARAM_NAMES, sobol_s1)),
    }


def _plot_uncertainty(table: pd.DataFrame, sobol_st: np.ndarray, output_path: str) -> None:
    """per-site P10/P50/P90 ranges (left) and the mean Sobol ST ranking (right)."""
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

        #left: P50 with P10-P90 whiskers, sites sorted by P50
        t = table.sort_values("p50").reset_index(drop=True)
        x = np.arange(len(t))
        labels = [f"{r.lat:.1f},{r.lon:.1f}" for r in t.itertuples()]
        lo = t["p50"] - t["p10"]
        hi = t["p90"] - t["p50"]
        ax[0].errorbar(x, t["p50"], yerr=[lo, hi], fmt="o", color="#1f77b4",
                       ecolor="#999", capsize=3, markersize=5)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels, rotation=90, fontsize=7)
        ax[0].set_ylabel("LCOE (GBP/MWh)")
        ax[0].set_title("Per-site LCOE: P50 with P10-P90 range")
        ax[0].grid(alpha=0.3, axis="y")

        #right: mean total-order sobol indices, ranked
        order = np.argsort(sobol_st)
        names = [_PARAM_NAMES[i] for i in order]
        ax[1].barh(np.arange(len(order)), sobol_st[order], color="#2ca02c")
        ax[1].set_yticks(np.arange(len(order)))
        ax[1].set_yticklabels(names, fontsize=8)
        ax[1].set_xlabel("Sobol total-order index ST (mean across sites)")
        ax[1].set_title("What drives LCOE variance")
        ax[1].grid(alpha=0.3, axis="x")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[UQ] Figure saved -> {output_path}")
    except Exception as e:
        print(f"[UQ] WARNING: Could not render figure: {e}")
