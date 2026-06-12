#src/economics.py - simplified lcoe model for candidate sites
#based on methodology from cavazzi & dutton

import numpy as np
import pandas as pd


def estimate_capex(depth_m: float, rated_power_mw: float, distance_km: float = 0.0) -> float:
    """
    estimate capex (£) for a single turbine using industry-standard component
    breakdowns, stepped foundation thresholds, and hvac/hvdc transmission logic.
    """
    #reference a 500 mw farm to allocate shared infrastructure (substation, export cable)
    farm_capacity_mw = 500.0
    number_of_turbines = farm_capacity_mw / rated_power_mw

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

    #4. export transmission - hvdc required beyond 80 km
    if distance_km <= 80:
        hvac_substation_cost = 50_000_000
        hvac_cable_cost = distance_km * 1_200_000
        total_transmission_cost = hvac_substation_cost + hvac_cable_cost
    else:
        hvdc_converter_stations = 250_000_000
        hvdc_cable_cost = distance_km * 800_000
        total_transmission_cost = hvdc_converter_stations + hvdc_cable_cost

    #allocate shared transmission cost to one turbine in the 500 mw reference farm
    transmission_capex_per_turbine = total_transmission_cost / number_of_turbines

    #5. installation and vessel day rates - small distance penalty for transit time
    installation_capex = (400_000 * rated_power_mw) + (distance_km * 5_000)

    return (turbine_capex +
            foundation_capex +
            internal_grid_capex +
            transmission_capex_per_turbine +
            installation_capex)


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
        distance = row.get("dist_to_port_km", 0.0)
        foundation_type = "floating" if depth > 60 else "fixed-bottom"

        capex = estimate_capex(depth, TURBINE["rated_power_mw"], distance)
        opex = capex * E["opex_pct_capex_per_year"]

        #adjust aep to account for array, electrical, and downtime losses if available
        aep_for_lcoe = row["aep_adjusted_mwh"]
        if "calculated_cf" in row and "mean_cf" in row and pd.notna(row["calculated_cf"]) and pd.notna(row["mean_cf"]) and row["mean_cf"] > 0:
            availability = TURBINE["availability"]
            current_ratio = aep_for_lcoe / (TURBINE["rated_power_mw"] * 8760 * row["mean_cf"])
            #if aep was only adjusted for availability, apply the additional loss factors
            if abs(current_ratio - availability) < 0.01:
                aep_for_lcoe = row["aep_adjusted_mwh"] * (row["calculated_cf"] / row["mean_cf"])

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

            depths = []
            for _, row in df.iterrows():
                elev = float(ds[elev_var].sel(
                    {lon_dim: row["lon"], lat_dim: row["lat"]},
                    method="nearest"
                ))
                depths.append(abs(elev) if elev < 0 else 0.0)
            df["depth_m"] = depths
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
        capex_turbine   = estimate_capex(depth, TURBINE["rated_power_mw"], distance_km=0.0)
        opex_turbine    = capex_turbine * E["opex_pct_capex_per_year"]
        lcoe_turbine    = (compute_lcoe(capex_turbine, aep_per_turbine, TURBINE["rated_power_mw"])
                           if aep_per_turbine > 0 else np.nan)

        #whole-farm lcoe (linear scale - no economies of scale captured)
        capex_per_turbine = estimate_capex(depth, TURBINE["rated_power_mw"], distance_km=0.0)
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