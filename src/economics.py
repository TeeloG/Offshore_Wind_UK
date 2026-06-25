#src/economics.py - simplified lcoe model for candidate sites
#based on methodology from cavazzi & dutton

import numpy as np
import pandas as pd


def reference_farm_summary() -> dict:
    """
    geometry of the standardised reference array used for cost amortisation.

    a square layout_side x layout_side array of identical turbines, spaced a
    given number of rotor diameters apart. footprint side = layout_side x
    spacing; shared transmission is amortised across all turbines.

    returns a dict with n_turbines, capacity_mw, spacing_m, footprint_side_km,
    footprint_km2.
    """
    from config import REFERENCE_FARM, TURBINE

    side = REFERENCE_FARM["layout_side"]
    n = side ** 2
    spacing_m = REFERENCE_FARM["spacing_rotor_diam"] * TURBINE["rotor_diameter_m"]
    footprint_side_km = side * spacing_m / 1000.0
    return {
        "n_turbines":        n,
        "capacity_mw":       n * TURBINE["rated_power_mw"],
        "spacing_m":         spacing_m,
        "footprint_side_km": footprint_side_km,
        "footprint_km2":     footprint_side_km ** 2,
    }


def estimate_capex(depth_m: float, rated_power_mw: float,
                   dist_to_port_km: float = 0.0,
                   dist_to_grid_km: float = None,
                   return_components: bool = False):
    """
    estimate capex (£) for a single turbine using industry-standard component
    breakdowns, stepped foundation thresholds, and hvac/hvdc transmission logic.

    two distances enter the model (bug ledger #8):
        dist_to_port_km: o&m / installation vessel transit to a maintenance port.
        dist_to_grid_km: export-cable length to the onshore grid-connection
                         point, approximated by distance to the nearest coast.
    if dist_to_grid_km is None it falls back to dist_to_port_km, preserving the
    old single-distance behaviour for callers that pass only one distance.

    if return_components is true, returns (total, components_dict) where the dict
    has per-turbine turbine/foundation/array/transmission/installation costs, for
    the monte carlo uncertainty module to perturb each component independently.
    """
    if dist_to_grid_km is None:
        dist_to_grid_km = dist_to_port_km

    #standardised reference array: shared transmission (substation/converter +
    #export cable) is amortised across the whole farm (block 2 phase 2).
    #default 10 x 10 x 15 mw = 1500 mw, replacing the old 500 mw (~33-turbine) ref.
    from config import REFERENCE_FARM
    number_of_turbines = REFERENCE_FARM["layout_side"] ** 2
    farm_capacity_mw = number_of_turbines * rated_power_mw

    #1. turbine cost (nacelle, rotor, tower) - roughly £1.2m/mw
    turbine_capex = 1_200_000 * rated_power_mw

    #2. foundation cost - stepped at real engineering limits
    if depth_m <= 35:
        foundation_capex = 800_000 * rated_power_mw    #monopile
    elif depth_m <= 60:
        foundation_capex = 1_300_000 * rated_power_mw  #jacket (significant steel/fabrication jump)
    else:
        foundation_capex = 3_000_000 * rated_power_mw  #floating (hull + mooring)

    #3. array cables and offshore substation
    internal_grid_capex = 500_000 * rated_power_mw

    #4. export transmission - export cable to shore; hvdc required beyond threshold.
    #   the substation/converter scales with farm capacity (£/mw) so the bigger
    #   array is not understated; the export cable scales with its length (£/km).
    from config import TRANSMISSION as T
    if dist_to_grid_km <= T["hvdc_threshold_km"]:
        substation_cost = T["hvac_substation_gbp_per_mw"] * farm_capacity_mw
        cable_cost = dist_to_grid_km * T["hvac_cable_gbp_per_km"]
        total_transmission_cost = substation_cost + cable_cost
    else:
        converter_cost = T["hvdc_converter_gbp_per_mw"] * farm_capacity_mw
        cable_cost = dist_to_grid_km * T["hvdc_cable_gbp_per_km"]
        total_transmission_cost = converter_cost + cable_cost

    #allocate the shared transmission cost to one turbine in the reference array
    transmission_capex_per_turbine = total_transmission_cost / number_of_turbines

    #5. installation and vessel day rates - small port-transit penalty for transit time
    installation_capex = (400_000 * rated_power_mw) + (dist_to_port_km * 5_000)

    total = (turbine_capex +
             foundation_capex +
             internal_grid_capex +
             transmission_capex_per_turbine +
             installation_capex)

    if return_components:
        components = {
            "turbine":      turbine_capex,
            "foundation":   foundation_capex,
            "array":        internal_grid_capex,
            "transmission": transmission_capex_per_turbine,
            "installation": installation_capex,
        }
        return total, components

    return total


def compute_lcoe(capex_gbp: float,
                 aep_mwh_per_year: float,
                 rated_power_mw: float) -> float:
    """
    compute levelised cost of energy (£/mwh) for a single turbine.

    lcoe = (capex × crf + opex) / aep
    crf  = r(1+r)^n / ((1+r)^n - 1)

    args:
        capex_gbp:         total upfront capital cost (£).
        aep_mwh_per_year:  adjusted annual energy production (mwh/turbine/yr).
        rated_power_mw:    turbine rated capacity (used to estimate opex).

    returns:
        lcoe in £/mwh.
    """
    from config import ECONOMICS as E

    n = E["project_lifetime_years"]
    r = E["discount_rate"]

    #capital recovery factor: converts lump-sum capex to an equivalent annual payment
    crf = r * (1 + r)**n / ((1 + r)**n - 1)

    #opex as a fixed % of capex per year
    opex_annual = capex_gbp * E["opex_pct_capex_per_year"]

    total_annual_cost = capex_gbp * crf + opex_annual

    if aep_mwh_per_year <= 0:
        return np.nan

    return round(total_annual_cost / aep_mwh_per_year, 2)


def run_economic_analysis(sites_with_aep: pd.DataFrame) -> pd.DataFrame:
    """
    run the full economic analysis pipeline for all candidate sites.

    expects a dataframe with columns: lat, lon, depth_m, aep_adjusted_mwh, mean_cf

    returns the same dataframe with capex, opex, and lcoe columns added.
    """
    from config import TURBINE, ECONOMICS as E

    farm = reference_farm_summary()
    print(f"[Economics] Reference array: {farm['n_turbines']} x {TURBINE['rated_power_mw']} MW "
          f"= {farm['capacity_mw']:.0f} MW, spacing {farm['spacing_m']:.0f} m, "
          f"footprint {farm['footprint_side_km']:.1f} x {farm['footprint_side_km']:.1f} km "
          f"({farm['footprint_km2']:.0f} km2).")

    df = sites_with_aep.copy()

    if "depth_m" not in df.columns:
        print("[Economics] WARNING: No depth_m column - using 40 m placeholder for all sites.")
        df["depth_m"] = 40.0

    if "aep_adjusted_mwh" not in df.columns:
        raise ValueError("Run energy analysis first - 'aep_adjusted_mwh' column missing.")

    capex_list      = []
    opex_list       = []
    lcoe_list       = []
    foundation_list = []

    for _, row in df.iterrows():
        depth = row["depth_m"]
        dist_port = row.get("dist_to_port_km", 0.0)
        #export cable runs to the nearest grid-connection point (proxied by the
        #coast), distinct from the o&m port transit distance (bug ledger #8).
        #fall back to the port distance if shore distance is unavailable.
        dist_grid = row.get("dist_to_shore_km", dist_port)
        if pd.isna(dist_grid):
            dist_grid = dist_port
        foundation_type = "floating" if depth > 60 else "fixed-bottom"

        capex = estimate_capex(depth, TURBINE["rated_power_mw"],
                               dist_to_port_km=dist_port, dist_to_grid_km=dist_grid)
        opex = capex * E["opex_pct_capex_per_year"]

        #aep fed to lcoe uses the post-loss capacity factor, applied exactly once.
        #calculated_cf already folds in array, electrical, and downtime losses, so
        #multiplying by availability (0.94) again would double-count the downtime
        #derate. availability and the 5.97% downtime are the same physical effect,
        #and applying both inflated every lcoe by ~6% (bug ledger #1). availability
        #is dropped from this path and kept only for the reported net aep column.
        if "calculated_cf" in row and pd.notna(row.get("calculated_cf")):
            aep_for_lcoe = TURBINE["rated_power_mw"] * 8760 * row["calculated_cf"]
        else:
            #no post-loss cf available, fall back to the reported net aep figure.
            aep_for_lcoe = row["aep_adjusted_mwh"]

        lcoe = compute_lcoe(capex, aep_for_lcoe, TURBINE["rated_power_mw"])

        capex_list.append(round(capex / 1e6, 3))    #£ -> £m
        opex_list.append(round(opex / 1e3, 1))      #£ -> £k/yr
        lcoe_list.append(lcoe)
        foundation_list.append(foundation_type)

    df["capex_gbp_millions"]   = capex_list
    df["opex_gbp_k_per_year"]  = opex_list
    df["lcoe_gbp_per_mwh"]     = lcoe_list
    df["foundation_type"]      = foundation_list

    print(
        "[Economics] Analysis complete. "
        f"LCOE range: GBP {df['lcoe_gbp_per_mwh'].min():.1f}-"
        f"GBP {df['lcoe_gbp_per_mwh'].max():.1f}/MWh"
    )
    return df


def compute_lcoe_for_v2_farms(v2_farms_df, gebco_nc_path: str = None):
    """
    calculate lcoe for the 44 real uk offshore wind farms from the v2 dataset.

    uses gebco to extract real water depth at each farm's coordinates, then
    runs the same lcoe model as candidate sites - allowing direct comparison.

    aep is calculated from each farm's total capacity and calculated_cf (which
    already accounts for array losses, electrical losses, and downtime).

    args:
        v2_farms_df:   dataframe from load_v2_wind_farms() - must have
                       lat, lon, capacity_mw, calculated_cf columns.
        gebco_nc_path: path to gebco netcdf. falls back to 40 m if missing.

    returns:
        v2_farms_df with depth_m, aep_mwh, foundation_type, capex_gbp_millions,
        opex_gbp_k_per_year, and lcoe_gbp_per_mwh columns added.
    """
    from config import ECONOMICS as E, DEPTH

    df = v2_farms_df.copy()

    #step 1: extract gebco depth
    if gebco_nc_path:
        try:
            import xarray as xr
            ds = xr.open_dataset(gebco_nc_path)
            elev_var = "elevation" if "elevation" in ds else list(ds.data_vars)[0]
            lon_dim  = "lon" if "lon" in ds.coords else "longitude"
            lat_dim  = "lat" if "lat" in ds.coords else "latitude"

            #vectorised nearest-neighbour lookup (bug ledger #4): one xr.sel with
            #array indexers instead of a per-row loop. identical result.
            lons = xr.DataArray(df["lon"].values, dims="pts")
            lats = xr.DataArray(df["lat"].values, dims="pts")
            elev = ds[elev_var].sel({lon_dim: lons, lat_dim: lats}, method="nearest").values
            df["depth_m"] = np.where(elev < 0, -elev, 0.0).astype(float)
            print(f"[Economics-V2] GEBCO depth extracted for {len(df)} farms. "
                  f"Range: {df['depth_m'].min():.0f}–{df['depth_m'].max():.0f} m")
        except Exception as e:
            print(f"[Economics-V2] WARNING: Could not extract GEBCO depth: {e}. "
                  "Using 40 m placeholder.")
            df["depth_m"] = 40.0
    else:
        print("[Economics-V2] No GEBCO path provided — using 40 m depth placeholder.")
        df["depth_m"] = 40.0

    #step 2: aep from real farm capacity and calculated cf
    #calculated_cf already includes all losses, so no further adjustment needed
    hours = 8760
    df["aep_mwh"] = df["capacity_mw"] * hours * df["calculated_cf"]

    #step 3: lcoe model
    #three lcoe figures per farm:
    #lcoe_per_turbine - per 15 mw turbine basis, comparable to candidate sites
    #lcoe_whole_farm  - whole farm capacity
    #lcoe_v2_flat     - flat £2.5m/mw capex for a data-driven cross-check
    from config import TURBINE

    capex_list, opex_list, foundation_list = [], [], []
    lcoe_turbine_list, lcoe_farm_list, lcoe_flat_list = [], [], []

    for _, row in df.iterrows():
        depth    = row["depth_m"]
        capacity = row["capacity_mw"]
        aep      = row["aep_mwh"]

        #use excel foundation type if available, otherwise infer from depth
        if "fixed_floating" in df.columns and pd.notna(row.get("fixed_floating")):
            ff_str = str(row["fixed_floating"]).strip().lower()
            foundation = "floating" if "float" in ff_str else "fixed-bottom"
        else:
            foundation = "floating" if depth > DEPTH["fixed_bottom_max_m"] else "fixed-bottom"

        #per-turbine lcoe (comparable with candidate sites)
        n_turbines      = capacity / TURBINE["rated_power_mw"]
        aep_per_turbine = aep / n_turbines if n_turbines > 0 else 0.0
        capex_turbine   = estimate_capex(depth, TURBINE["rated_power_mw"],
                                          dist_to_port_km=0.0, dist_to_grid_km=0.0)
        opex_turbine    = capex_turbine * E["opex_pct_capex_per_year"]
        lcoe_turbine    = (compute_lcoe(capex_turbine, aep_per_turbine, TURBINE["rated_power_mw"])
                           if aep_per_turbine > 0 else np.nan)

        #whole-farm lcoe (linear scale - no economies of scale captured)
        capex_per_turbine = estimate_capex(depth, TURBINE["rated_power_mw"],
                                            dist_to_port_km=0.0, dist_to_grid_km=0.0)
        capex_farm = capex_per_turbine * n_turbines
        opex_farm  = capex_farm * E["opex_pct_capex_per_year"]
        lcoe_farm  = compute_lcoe(capex_farm, aep, capacity) if aep > 0 else np.nan

        #v2 flat capex lcoe (£2.5m/mw, data-driven)
        flat_capex_per_mw = 2.5e6
        capex_flat = capacity * flat_capex_per_mw
        opex_flat  = capex_flat * E["opex_pct_capex_per_year"]
        lcoe_flat  = compute_lcoe(capex_flat, aep, capacity) if aep > 0 else np.nan

        capex_list.append(round(capex_turbine / 1e6, 3))
        opex_list.append(round(opex_turbine / 1e3, 1))
        lcoe_turbine_list.append(lcoe_turbine)
        lcoe_farm_list.append(lcoe_farm)
        lcoe_flat_list.append(lcoe_flat)
        foundation_list.append(foundation)

    df["capex_gbp_millions"]          = capex_list
    df["opex_gbp_k_per_year"]         = opex_list
    df["lcoe_gbp_per_mwh"]            = lcoe_turbine_list
    df["lcoe_whole_farm_gbp_per_mwh"] = lcoe_farm_list
    df["lcoe_v2_flat_gbp_per_mwh"]    = lcoe_flat_list
    df["foundation_type"]             = foundation_list

    valid = df["lcoe_gbp_per_mwh"].dropna()
    print(f"[Economics-V2] Per-turbine LCOE for {len(valid)}/{len(df)} farms. "
          f"Range: £{valid.min():.1f}–£{valid.max():.1f}/MWh")
    valid_farm = df["lcoe_whole_farm_gbp_per_mwh"].dropna()
    print(f"[Economics-V2] Whole-farm LCOE for {len(valid_farm)}/{len(df)} farms. "
          f"Range: £{valid_farm.min():.1f}–£{valid_farm.max():.1f}/MWh")
    valid_flat = df["lcoe_v2_flat_gbp_per_mwh"].dropna()
    print(f"[Economics-V2] V2-flat LCOE for {len(valid_flat)}/{len(df)} farms. "
          f"Range: £{valid_flat.min():.1f}–£{valid_flat.max():.1f}/MWh")
    return df