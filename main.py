#!/usr/bin/env python3
"""
main.py - offshore wind energy assessment pipeline (v1)
tyler nolan green

run this file to execute the full pipeline:
    python main.py

phases:
    1. data collection & preprocessing
    2. spatial analysis & suitability mapping
    3. energy yield estimation (aep)
    4. economic analysis (lcoe)
    5. visualisation (folium interactive maps + matplotlib static figures)
"""

import sys
import os
import time
import webbrowser
from pathlib import Path
import pandas as pd

#add the project root so imports from config.py and src/ work when running main.py directly.
sys.path.insert(0, os.path.dirname(__file__))

from config import EEZ_SHAPEFILE, NINJA_API, TURBINE

#data loading
from src.data_loader import (
    make_placeholder_wind_grid,
    load_era5_wind,
    add_gebco_depth,
    add_port_distances,
    load_marine_protected_areas,
)

#spatial analysis
from src.suitability import (
    apply_hard_exclusions,
    score_soft_constraints,
    select_candidate_sites,
    precompute_all_combinations,
)

#energy & economics
from src.energy import compute_aep_for_all_sites
from src.economics import run_economic_analysis, compute_lcoe_for_v2_farms

#visualisation (interactive html)
from src.visualisation import (
    make_wind_resource_map,
    make_suitability_map,
    make_capacity_factor_chart,
    make_lcoe_chart,
    make_cf_timeseries_chart,
)

#visualisation (static png for dissertation)
from src.visualisation import (
    make_static_wind_map,
    make_static_suitability_map,
    make_static_lcoe_chart,
    make_static_cf_chart,
)

#resolve eez shapefile path
#path is relative to the project root in config.py.
_PROJECT_ROOT = Path(__file__).resolve().parent
_EEZ_RESOLVED = None
if EEZ_SHAPEFILE:
    _eez_candidate = _PROJECT_ROOT / EEZ_SHAPEFILE
    if _eez_candidate.is_file():
        _EEZ_RESOLVED = str(_eez_candidate)
    else:
        print(
            f"[Config] EEZ shapefile not found at {_eez_candidate} "
            f"(need .shp alongside .dbf/.shx). Using full UK bbox; no EEZ outline."
        )

#gebco bathymetry path
#download from https://download.gebco.net - uk bounding box, 15 arc-second, netcdf.
GEBCO_NC_PATH = str(_PROJECT_ROOT / "data" / "raw" / "gebco" / "gebco_2025_uk.nc")

#marine protected areas shapefile path
#download from https://jncc.gov.uk/our-work/uk-marine-protected-area-datasets-for-download/
MPA_SHAPEFILE = str(_PROJECT_ROOT / "data" / "raw" / "mpa" / "OffshoreMPA.shp")


#=============================================================================
#phase 1 - data loading
#=============================================================================
print("\n" + "="*60)
print("  PHASE 1 - Data Loading")
print("="*60)

#synthetic wind grid clipped to uk waters. bounding box was expanded after
#the original limits cut off northern scotland and the atlantic margin.
wind_grid = make_placeholder_wind_grid(
    resolution_deg=0.5,
    eez_shapefile=_EEZ_RESOLVED,
)
print(f"  Wind grid: {len(wind_grid)} points loaded.")

#add real gebco depths so the lcoe model uses actual seabed conditions.
#falls back to 40 m if the bathymetry file is missing.
wind_grid = add_gebco_depth(wind_grid, GEBCO_NC_PATH)

print(f"  Wind speed stats — mean: {wind_grid['mean_wind_ms'].mean():.1f} m/s, "
      f"min: {wind_grid['mean_wind_ms'].min():.1f} m/s, "
      f"max: {wind_grid['mean_wind_ms'].max():.1f} m/s")

wind_grid = add_port_distances(wind_grid)

print(f"  Wind grid after enrichment: {len(wind_grid)} points, "
      f"columns: {list(wind_grid.columns)}\n")

#shapefiles
from src.data_loader import load_marine_protected_areas, load_seabed_substrate

#bgs 250k seabed sediments - real seabed type data for the soft constraint score
BGS_SEABED_PATH = str(_PROJECT_ROOT / "data" / "raw" / "bgs_seabed" /
                      "BGS_250k_SeaBedSediments_WGS84_v3.shp")
seabed_gdf = load_seabed_substrate(BGS_SEABED_PATH)

mpa_gdf = load_marine_protected_areas(MPA_SHAPEFILE) if os.path.exists(MPA_SHAPEFILE) else None
if mpa_gdf is None:
    print(f"[Info] MPA file not found at {MPA_SHAPEFILE} — skipping MPA filtering.")

#v2 wind farm data
#real renewables.ninja cfs for 44 operational uk farms; replaces the synthetic fallback.
V2_DATA_PATH = str(_PROJECT_ROOT / "data" / "raw" / "v2" / "Offshore wind farms.xlsx")
from src.data_loader import load_v2_wind_farms, load_v2_default_losses
v2_farms = load_v2_wind_farms(V2_DATA_PATH)
v2_losses = load_v2_default_losses(V2_DATA_PATH)

#run lcoe on real farm locations for comparison with candidate site results
if v2_farms is not None:
    v2_farms = compute_lcoe_for_v2_farms(v2_farms, gebco_nc_path=GEBCO_NC_PATH)


#=============================================================================
#phase 2 - spatial analysis & suitability mapping
#=============================================================================
print("="*60)
print("  PHASE 2 - Spatial Analysis & Suitability Mapping")
print("="*60)

grid = apply_hard_exclusions(wind_grid, mpa_gdf=mpa_gdf)

#score across four soft constraints and compute weighted composite.
#port distances are now real values from phase 1, so the 20% weight is meaningful.
grid = score_soft_constraints(grid, seabed_gdf=seabed_gdf)

#select top 15 sites with minimum 1.0° spatial separation
candidate_sites = select_candidate_sites(grid, top_n=15)
print(f"  Candidate sites selected: {len(candidate_sites)}\n")

#pre-compute top 15 sites for every valid parameter combination and both mpa modes.
#stored as a nested dict for the html toggle panel - no reload needed to switch.
print("  Pre-computing all parameter combinations for the map toggle...")
site_combinations = precompute_all_combinations(grid, top_n=15)
print()


#=============================================================================
#phase 3 - energy yield estimation
#=============================================================================
print("="*60)
print("  PHASE 3 - Energy Yield Estimation (AEP)")
print("="*60)

#year 2019 avoids covid distortions and matches the available era5/api dataset.
sites_with_aep = compute_aep_for_all_sites(
    candidate_sites,
    ninja_token=NINJA_API["token"],
    year=NINJA_API["year"],
    v2_farms_df=v2_farms,
)
print(f"  AEP computed for {len(sites_with_aep)} sites.\n")

#apply v2-derived loss factors to get a more realistic post-loss cf
from src.energy import apply_losses_to_cf

sites_with_aep["calculated_cf"] = sites_with_aep["mean_cf"].apply(
    lambda cf: apply_losses_to_cf(
        cf,
        array_loss_pct=v2_losses["array_loss_pct"],
        electrical_loss_pct=v2_losses["electrical_loss_pct"],
        downtime_pct=v2_losses["downtime_pct"]
    )
)

#dunkelflaute detection on each candidate site
from src.energy import flag_dunkelflaute
from src.data_loader import fetch_ninja_capacity_factors

for idx, site in sites_with_aep.iterrows():
    cf_series = fetch_ninja_capacity_factors(
        lat=site["lat"],
        lon=site["lon"],
        token=NINJA_API["token"],
        year=NINJA_API["year"],
    )
    events = flag_dunkelflaute(cf_series)
    print(f"\nSite {idx} ({site['lat']:.1f}, {site['lon']:.1f}): "
          f"{len(events)} Dunkelflaute events")

for idx, site in sites_with_aep.iterrows():
    cf_series = fetch_ninja_capacity_factors(
        lat=site["lat"],
        lon=site["lon"],
        token=NINJA_API["token"],
        year=NINJA_API["year"],
    )
    events = flag_dunkelflaute(cf_series)
    if len(events) > 0:
        print(f"\nSite {idx} ({site['lat']:.1f}, {site['lon']:.1f}):")
        print(events.to_string(index=False))


#=============================================================================
#phase 4 - economic analysis (lcoe)
#=============================================================================
print("="*60)
print("  PHASE 4 - Economic Analysis (LCOE)")
print("="*60)

sites_final = run_economic_analysis(sites_with_aep)
sites_final_original = sites_final.copy()  #keep a clean copy for static charts

#expand sites_final to cover all sites from parameter combinations
#the economic lookup only has the original 15 phase 4 sites. sites that appear
#under toggled parameters haven't been through phase 3/4, so we estimate their
#cf from wind speed data to avoid additional api calls, then run economics.
from src.energy import estimate_cf_from_wind_speed
from src.economics import run_economic_analysis
from config import TURBINE

all_sites_coords = set()
for branch in site_combinations.values():
    for combo_sites in branch.values():
        for site in combo_sites:
            all_sites_coords.add((round(site["lat"], 4), round(site["lon"], 4)))

existing_coords = set((round(row["lat"], 4), round(row["lon"], 4)) for _, row in sites_final.iterrows())
missing_coords = all_sites_coords - existing_coords

if missing_coords:
    print(f"[Main] Found {len(missing_coords)} additional sites in parameter combinations. "
          "Estimating energy from wind speed data...")

    missing_sites = []
    for lat, lon in missing_coords:
        match = grid[(grid["lat"].round(4) == lat) & (grid["lon"].round(4) == lon)]
        if not match.empty:
            missing_sites.append(match.iloc[0])

    if missing_sites:
        missing_df = pd.DataFrame(missing_sites).copy()

        missing_df["mean_cf"] = missing_df["mean_wind_ms"].apply(
            lambda ws: estimate_cf_from_wind_speed(ws, TURBINE["rated_power_mw"])
        )

        from src.energy import apply_losses_to_cf
        missing_df["calculated_cf"] = missing_df["mean_cf"].apply(
            lambda cf: apply_losses_to_cf(
                cf,
                array_loss_pct=v2_losses["array_loss_pct"],
                electrical_loss_pct=v2_losses["electrical_loss_pct"],
                downtime_pct=v2_losses["downtime_pct"]
            )
        )

        missing_df["aep_adjusted_mwh"] = (
            TURBINE["rated_power_mw"] * 8760 * missing_df["calculated_cf"]
        )

        missing_final = run_economic_analysis(missing_df)
        sites_final = pd.concat([sites_final, missing_final], ignore_index=True)
        print(f"[Main] Added {len(missing_final)} sites with estimated energy data. "
              f"Total sites: {len(sites_final)}")
else:
    print("[Main] All sites from parameter combinations already have economic data.")

print("\n--- Full 15-site Results Table ---")
print(sites_final[["lat", "lon", "depth_m", "foundation_type",
                    "mean_cf", "calculated_cf",
                    "aep_adjusted_mwh", "capex_gbp_millions",
                    "lcoe_gbp_per_mwh"]].to_string(index=True))


#=============================================================================
#phase 5 - visualisation
#=============================================================================
print("="*60)
print("  PHASE 5 - Visualisation")
print("="*60)

#interactive html outputs
map1   = make_wind_resource_map(wind_grid, eez_shapefile=_EEZ_RESOLVED)
map2   = make_suitability_map(grid, candidate_sites=sites_final,
                               eez_shapefile=_EEZ_RESOLVED, v2_farms_df=v2_farms,
                               site_combinations=site_combinations,
                               sites_final_df=sites_final)
chart1 = make_capacity_factor_chart(sites_final)
chart2 = make_lcoe_chart(sites_final)

#static png outputs
print("\n  Generating static dissertation figures ...")

fig1 = make_static_wind_map(wind_grid, eez_shapefile=_EEZ_RESOLVED)
fig2 = make_static_suitability_map(
    grid,
    candidate_sites=sites_final_original,
    eez_shapefile=_EEZ_RESOLVED,
    v2_farms_df=v2_farms,
)
fig3 = make_static_lcoe_chart(sites_final_original)
fig4 = make_static_cf_chart(sites_final_original)

print("\n" + "="*60)
print("  OK  Pipeline complete. Outputs:")
print(f"     Interactive maps:   outputs/maps/")
print(f"     Interactive charts: outputs/charts/")
print(f"     Static figures:     outputs/maps/  and  outputs/charts/")
print("="*60 + "\n")


#=============================================================================
#open interactive maps in browser
#=============================================================================

def _open_html_in_browser(path: str, label: str) -> None:
    """
    open a local html file in the system's default browser.
    uses os.startfile() on windows - more reliable for multiple local files.
    """
    p = Path(path).resolve()
    uri = p.as_uri()
    print(f"  Opening {label}: {uri}")

    if sys.platform == "win32":
        try:
            os.startfile(str(p))
        except OSError as exc:
            print(f"  (startfile failed: {exc}; trying webbrowser)")
            if not webbrowser.open(uri):
                print("  (Could not open browser — open the file manually.)")
    else:
        if not webbrowser.open(uri):
            print("  (Could not open browser — open the file manually.)")


_open_html_in_browser(map1, "wind resource map")
time.sleep(1.0)
_open_html_in_browser(map2, "suitability map")