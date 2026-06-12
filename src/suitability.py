#src/suitability.py - spatial exclusions and weighted suitability scoring.
#hard constraints remove infeasible locations entirely; soft constraints
#score remaining points on a 0-1 scale for comparative site ranking.

import numpy as np
import pandas as pd


#hard constraint mask

def apply_hard_exclusions(grid_df: pd.DataFrame,
                           mpa_gdf=None,
                           shipping_lanes_gdf=None,
                           cables_gdf=None) -> pd.DataFrame:
    """
    apply binary exclusion mask for hard constraints.
    any point inside a hard exclusion zone gets excluded=true.

    hard exclusions (from literature review):
        - marine protected areas (jncc / natural england)
        - shipping lanes
        - oil & gas safety zones
        - existing submarine cables

    args:
        grid_df:            dataframe with 'lon', 'lat' columns
        mpa_gdf:            geodataframe of mpa polygons (or none)
        shipping_lanes_gdf: geodataframe of shipping lane polygons
        cables_gdf:         geodataframe of cable exclusion zones

    returns:
        grid_df with 'hard_excluded', 'hard_excluded_nonmpa', 'inside_mpa',
        and 'exclusion_reason' columns added.

    todo: implement spatial joins for shipping lanes and cables once shapefiles
          are available.
    """
    grid_df = grid_df.copy()

    #start with all points included; flip to true as each constraint is applied
    grid_df["hard_excluded"] = False
    grid_df["hard_excluded_nonmpa"] = False
    grid_df["inside_mpa"] = False
    grid_df["exclusion_reason"] = None

    #mpa exclusion
    if mpa_gdf is not None:
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            points_gdf = gpd.GeoDataFrame(
                grid_df,
                geometry=[Point(lon, lat) for lon, lat in zip(grid_df["lon"], grid_df["lat"])],
                crs="EPSG:4326"
            )

            #how="left" preserves all grid points; index_right notna means inside an mpa
            inside_mpa = gpd.sjoin(points_gdf, mpa_gdf, how="left", predicate="within")
            excluded_indices = inside_mpa[inside_mpa["index_right"].notna()].index

            #mpa points keep their scores so they can appear when the toggle is off
            grid_df.loc[excluded_indices, "inside_mpa"] = True
            grid_df.loc[excluded_indices, "hard_excluded"] = True
            grid_df.loc[excluded_indices, "exclusion_reason"] = "Inside Marine Protected Area"

            n_excluded = len(excluded_indices)
            print(f"[Suitability] {n_excluded} points excluded: inside Marine Protected Area")

        except ImportError:
            print("[Suitability] WARNING: geopandas not installed. MPA exclusion skipped.")
        except Exception as e:
            print(f"[Suitability] WARNING: MPA spatial join failed: {e}")

    #shipping lane exclusion
    if shipping_lanes_gdf is not None:
        #todo: spatial join - mark points inside major shipping corridors
        pass

    #cable / pipeline exclusion
    if cables_gdf is not None:
        #todo: spatial join with buffer (~500 m clearance either side)
        pass

    #water depth
    from config import DEPTH

    if "depth_m" in grid_df.columns:
        shallow_mask = grid_df["depth_m"] < 10
        grid_df.loc[shallow_mask, ["hard_excluded", "hard_excluded_nonmpa"]] = True
        grid_df.loc[shallow_mask, "exclusion_reason"] = "Too shallow (< 10 m). Risk of grounding and navigation hazards"

        deep_mask = grid_df["depth_m"] > DEPTH["floating_max_m"]
        grid_df.loc[deep_mask, ["hard_excluded", "hard_excluded_nonmpa"]] = True
        grid_df.loc[deep_mask, "exclusion_reason"] = f"Too deep (> {DEPTH['floating_max_m']} m). Beyond the feasible limit for floating turbine technology"

    total_excluded = grid_df["hard_excluded"].sum()
    nonmpa_excluded = grid_df["hard_excluded_nonmpa"].sum()
    mpa_only = ((grid_df["hard_excluded"]) & (~grid_df["hard_excluded_nonmpa"])).sum()
    print(
        f"[Suitability] Hard exclusions applied: {total_excluded}/{len(grid_df)} points excluded "
        f"({nonmpa_excluded} non-MPA, {mpa_only} MPA-only)."
    )
    return grid_df


#soft constraint scoring

def score_soft_constraints(grid_df: pd.DataFrame,
                            seabed_gdf=None) -> pd.DataFrame:
    """
    score remaining (non-excluded) points on a 0-1 scale for each soft constraint,
    then compute a weighted composite suitability score.

    soft constraints and weights (from config.py):
        wind_resource (0.40), water_depth (0.25),
        distance_to_port (0.20), seabed_type (0.15)

    args:
        grid_df:    scored grid dataframe from apply_hard_exclusions.
        seabed_gdf: optional bgs seabed geodataframe. if provided, each grid point
                    is spatially joined to the nearest sediment polygon.

    returns:
        grid_df with score_wind, score_depth, score_port, score_seabed,
        and suitability_score columns added.
    """
    from config import SOFT_WEIGHTS, DEPTH

    df = grid_df.copy()

    #spatial join: attach bgs seabed type
    #points outside the data boundary get nan and fall back to 0.5 below.
    if seabed_gdf is not None and "seabed_type" not in df.columns:
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            points_gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                crs="EPSG:4326"
            )
            joined = gpd.sjoin(
                points_gdf,
                seabed_gdf[["seabed_type", "geometry"]],
                how="left",
                predicate="within"
            )
            #deduplicate if a point touches multiple polygons
            joined = joined[~joined.index.duplicated(keep="first")]
            df["seabed_type"] = joined["seabed_type"].values
            n_matched = df["seabed_type"].notna().sum()
            print(f"[Suitability] BGS seabed join: {n_matched}/{len(df)} points matched.")
        except Exception as e:
            print(f"[Suitability] WARNING: Seabed spatial join failed: {e}. "
                  "Using 0.5 placeholder for seabed score.")

    #only score points not excluded for non-mpa reasons. mpa points retain
    #scores so they can be shown when the mpa toggle is off.
    candidates = df[~df["hard_excluded_nonmpa"]].copy()

    #wind resource score
    if "mean_wind_ms" in candidates.columns:
        ws = candidates["mean_wind_ms"]
        #min-max normalise; 1e-9 prevents division by zero if all speeds identical
        candidates["score_wind"] = (ws - ws.min()) / (ws.max() - ws.min() + 1e-9)
    else:
        candidates["score_wind"] = 0.5

    #water depth score
    #sweet spot: 20-60 m (fixed-bottom monopile). scores fall off either side.
    if "depth_m" in candidates.columns:
        depth = candidates["depth_m"]
        candidates["score_depth"] = np.where(
            depth.between(20, DEPTH["fixed_bottom_max_m"]), 1.0,
            np.where(
                depth < 20,
                depth / 20,   #linear ramp 0 -> 1 as depth goes 0 -> 20 m
                np.clip(1 - (depth - DEPTH["fixed_bottom_max_m"]) / 240, 0, 1)
                #linear decline 1 -> 0 as depth goes 60 -> 300 m
            )
        )
    else:
        candidates["score_depth"] = 0.5

    #distance to port score
    if "dist_to_port_km" in candidates.columns:
        d = candidates["dist_to_port_km"]
        #invert-normalise: closest port = 1.0, furthest = 0.0
        candidates["score_port"] = np.clip(1 - d / d.max(), 0, 1)
    else:
        candidates["score_port"] = 0.5

    #seabed type score
    #sandy/gravelly sediments score highest
    #rock and diamicton score lowest
    SEABED_SCORES = {
        #pure sand and gravel - ideal
        "sand":                                                    1.0,
        "gravel":                                                  0.9,
        "sandy gravel":                                            0.9,
        "gravelly sand":                                           0.9,
        "gravel (french classification)":                          0.9,
        "gravelly sand [french classification]":                   0.85,
        "sandy gravel (french classification)":                    0.85,
        "slightly gravelly sand":                                  0.95,
        #mixed sediments - workable
        "gravel, sand and silt":                                   0.7,
        "gravelly muddy sand":                                     0.7,
        "muddy sandy gravel":                                      0.65,
        "slightly gravelly muddy sand":                            0.65,
        "mussel deposit":                                          0.7,
        #mud-dominated - requires scour protection
        "mud":                                                     0.5,
        "muddy sand":                                              0.55,
        "sandy mud":                                               0.5,
        "gravelly mud":                                            0.55,
        "muddy gravel":                                            0.6,
        "slightly gravelly mud":                                   0.5,
        "slightly gravelly sandy mud":                             0.5,
        #rock and diamicton - highest capex impact
        "rock (undifferentiated)":                                 0.3,
        "rock or diamicton (sea bed sediment, based on folk)":     0.3,
        "diamicton":                                               0.35,
    }

    if "seabed_type" in candidates.columns:
        candidates["score_seabed"] = (
            candidates["seabed_type"]
            .str.lower()
            .str.strip()
            .map(SEABED_SCORES)
            .fillna(0.5)   #neutral score for unrecognised types
        )
    else:
        candidates["score_seabed"] = 0.5

    #weighted composite score
    w = SOFT_WEIGHTS
    candidates["suitability_score"] = (
        w["wind_resource"]    * candidates["score_wind"]   +
        w["water_depth"]      * candidates["score_depth"]  +
        w["distance_to_port"] * candidates["score_port"]   +
        w["seabed_type"]      * candidates["score_seabed"]
    )

    #write scores back to the full grid, including mpa-only excluded points,
    #so they show up correctly when the mpa toggle is disabled on the map.
    for col in ["score_wind", "score_depth", "score_port",
                "score_seabed", "suitability_score"]:
        df.loc[~df["hard_excluded_nonmpa"], col] = candidates[col].values

    print(f"[Suitability] Scored {len(candidates)} candidate points. "
          f"Mean suitability: {candidates['suitability_score'].mean():.3f}")
    return df


#identify top candidate sites

def select_candidate_sites(scored_df: pd.DataFrame,
                            top_n: int = 20,
                            min_separation_deg: float = 1.0,
                            exclude_mpa: bool = True) -> pd.DataFrame:
    """
    select the top-n candidate sites by suitability score, enforcing a minimum
    spatial separation to avoid clustering.

    if exclude_mpa is false, points inside mpas but not otherwise excluded
    are eligible for selection.

    returns a dataframe of candidate sites for energy and economic analysis.
    """
    mask = (~scored_df["hard_excluded"]
            if exclude_mpa
            else ~scored_df["hard_excluded_nonmpa"])

    candidates = (
        scored_df[mask]
        .dropna(subset=["suitability_score"])
        .sort_values("suitability_score", ascending=False)
        .reset_index(drop=True)
    )

    #greedy selection with minimum spatial separation.
    #1.0° buffer keeps sites roughly 65-111 km apart without full geodesic overhead.
    selected = []

    for _, row in candidates.iterrows():
        if len(selected) >= top_n:
            break

        too_close = any(
            abs(row["lat"] - s["lat"]) < min_separation_deg and
            abs(row["lon"] - s["lon"]) < min_separation_deg
            for s in selected
        )

        if not too_close:
            selected.append(row)

    result = pd.DataFrame(selected).reset_index(drop=True)
    result.index.name = "site_id"

    print(f"[Suitability] Selected {len(result)} candidate sites.")
    return result


#interactive parameter toggle

def recompute_candidate_sites(scored_grid: pd.DataFrame,
                               active_params: list,
                               top_n: int = 15,
                               min_separation_deg: float = 1.0,
                               exclude_mpa: bool = True) -> pd.DataFrame:
    """
    recompute composite suitability scores using only the selected parameters,
    then re-select the top candidate sites.

    individual scores (score_wind, score_depth, score_port, score_seabed) must
    already exist in scored_grid - only the weighting changes here, so results
    are instant without re-scoring.

    disabled parameters have their weights redistributed proportionally across
    the remaining active parameters so the composite always sums to 1.0.

    args:
        scored_grid:        full grid dataframe from score_soft_constraints().
        active_params:      list of parameter keys to include. valid keys:
                            "wind_resource", "water_depth", "distance_to_port",
                            "seabed_type". at least one must be provided.
        top_n:              number of candidate sites to return (default 15).
        min_separation_deg: minimum spatial separation in degrees.
        exclude_mpa:        if false, mpa sites not otherwise excluded are eligible.

    returns:
        dataframe of top-n candidate sites with 'suitability_score' column
        reflecting the toggled weights.
    """
    from config import SOFT_WEIGHTS

    PARAM_TO_COL = {
        "wind_resource":    "score_wind",
        "water_depth":      "score_depth",
        "distance_to_port": "score_port",
        "seabed_type":      "score_seabed",
    }

    if not active_params:
        raise ValueError("At least one parameter must be active to recompute suitability.")

    active_weights = {p: SOFT_WEIGHTS[p] for p in active_params if p in SOFT_WEIGHTS}

    #renormalise so remaining weights still sum to 1.0
    total = sum(active_weights.values())
    normalised = {p: w / total for p, w in active_weights.items()}

    print(f"[Suitability] Recomputing with params: {list(normalised.keys())}")
    print(f"  Renormalised weights: { {p: round(w, 3) for p, w in normalised.items()} }")

    mask = (~scored_grid["hard_excluded"]
            if exclude_mpa
            else ~scored_grid["hard_excluded_nonmpa"])

    df = scored_grid[mask].copy()

    df["suitability_score"] = sum(
        normalised[p] * df[PARAM_TO_COL[p]]
        for p in active_params
        if p in PARAM_TO_COL and PARAM_TO_COL[p] in df.columns
    )

    #same greedy spatial-separation selection as select_candidate_sites()
    candidates = (
        df.dropna(subset=["suitability_score"])
        .sort_values("suitability_score", ascending=False)
        .reset_index(drop=True)
    )

    selected = []
    for _, row in candidates.iterrows():
        if len(selected) >= top_n:
            break
        too_close = any(
            abs(row["lat"] - s["lat"]) < min_separation_deg and
            abs(row["lon"] - s["lon"]) < min_separation_deg
            for s in selected
        )
        if not too_close:
            selected.append(row)

    result = pd.DataFrame(selected).reset_index(drop=True)
    result.index.name = "site_id"

    print(f"[Suitability] Recomputed: {len(result)} sites selected.")
    return result


def precompute_all_combinations(scored_grid: pd.DataFrame,
                                 top_n: int = 15,
                                 min_separation_deg: float = 1.0) -> dict:
    """
    pre-compute top candidate sites for every valid parameter combination.

    with 4 parameters there are 2^4 - 1 = 15 non-empty combinations, computed
    here once so the interactive html map can switch between them client-side
    without any further python execution.

    args:
        scored_grid:        full scored grid from score_soft_constraints().
        top_n:              number of sites per combination (default 15).
        min_separation_deg: spatial separation threshold.

    returns:
        dict with 'mpa_excluded' and 'mpa_allowed' branches, each mapping
        sorted comma-joined param keys to lists of site dicts containing:
        lat, lon, rank, score, depth_m, foundation_type.
    """
    from itertools import combinations as itercombs

    ALL_PARAMS = ["wind_resource", "water_depth", "distance_to_port", "seabed_type"]
    result = {"mpa_excluded": {}, "mpa_allowed": {}}

    total = 2 * sum(1 for r in range(1, len(ALL_PARAMS) + 1)
                    for _ in itercombs(ALL_PARAMS, r))
    done = 0

    for mpa_excluded in [True, False]:
        branch_key = "mpa_excluded" if mpa_excluded else "mpa_allowed"
        for r in range(1, len(ALL_PARAMS) + 1):
            for combo in itercombs(ALL_PARAMS, r):
                combo = list(combo)
                #alphabetically sorted key so js lookup is unambiguous
                key = ",".join(sorted(combo))
                try:
                    sites_df = recompute_candidate_sites(
                        scored_grid, combo,
                        top_n=top_n,
                        min_separation_deg=min_separation_deg,
                        exclude_mpa=mpa_excluded,
                    )
                    #store only the fields the map js needs
                    result[branch_key][key] = [
                        {
                            "lat":        round(float(row["lat"]), 4),
                            "lon":        round(float(row["lon"]), 4),
                            "rank":       int(i + 1),
                            "score":      round(float(row.get("suitability_score", 0)), 3),
                            "depth":      (round(float(row["depth_m"]), 0)
                                           if pd.notna(row.get("depth_m")) else None),
                            "foundation": ("floating" if row.get("depth_m", 0) > 60 else "fixed-bottom"),
                        }
                        for i, (_, row) in enumerate(sites_df.iterrows())
                    ]
                except Exception as exc:
                    print(f"[Suitability] WARNING: combo '{branch_key}/{key}' failed: {exc}")
                    result[branch_key][key] = []
                done += 1

    print(f"[Suitability] Precomputed {done}/{total} parameter combinations for MPA excluded/allowed toggles.")
    return result