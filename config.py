#config.py  centralised assumptions for the offshore wind pipeline.
#adjust values here to change the study area, turbine model, depth limits,
#and economic parameters without touching the analysis code.

#geographic scope
#uk_bbox is the initial rectangular grid envelope before the eez clip.
#expanded to fully contain the eez polygon - a 0.1° buffer avoids losing
#points on the boundary.
UK_BBOX = {
    "min_lon": -15.0,   #eez min_lon = -14.90°  (+0.1° buffer)
    "max_lon":   3.6,   #eez max_lon =   3.40°
    "min_lat":  47.3,   #eez min_lat =  47.44°
    "max_lat":  64.0,   #eez max_lat =  63.89°  (main culprit for missing n scotland)
}

#uk eez polygon (path relative to project root)
EEZ_SHAPEFILE = "data/raw/eez/eez.shp"

#named reference locations for map markers
REGIONS_OF_INTEREST = {
    "North Sea (East Scotland)":   [  0.5, 57.5],
    "North Sea (East England)":    [  1.5, 54.5],
    "Irish Sea (N of Anglesey)":   [ -4.5, 53.8],
    "North Atlantic (N Scotland)": [ -5.0, 59.5],
    "English Channel":             [ -2.0, 50.5],
}

#turbine assumptions
TURBINE = {
    "rated_power_mw":   15,
    "hub_height_m":    120,
    "rotor_diameter_m": 236,     #vestas v236-15.0
    "availability":     0.94,    #operational availability (crabtree et al. 2015)
}

#water depth constraints
DEPTH = {
    "fixed_bottom_max_m":   60,   #monopile/jacket limit
    "floating_min_m":       60,
    "floating_max_m":      300,   #practical floating limit
    "excluded_below_m":      0,
}

#site selection
#minimum great-circle separation between selected candidate sites (km).
#replaces the old anisotropic 1 degree box test (bug ledger #5); a degree box
#spans ~72 km east-west but ~111 km north-south at uk latitudes, so it enforced
#an uneven separation. haversine km is isotropic.
#100 km is chosen on methodological grounds: it keeps the chosen sites in
#genuinely distinct development zones and promotes the geographic decorrelation
#of low-wind (dunkelflaute) events that the portfolio argument relies on. it is
#set independently of which points happen to have cached capacity-factor data.
SITE_SELECTION = {
    "min_separation_km": 100.0,
}

#soft constraint weights
#must sum to 1.0 so the composite score stays on a 0-1 scale.
SOFT_WEIGHTS = {
    "wind_resource":    0.40,
    "water_depth":      0.25,
    "distance_to_port": 0.20,
    "seabed_type":      0.15,
}

#soft-score physical anchors (bug ledger #6)
#fixed physical bands so a site's wind and port scores are comparable across
#runs and regions, instead of min-max normalisation that shifts whenever the
#grid or candidate set changes. these bands are set on physical grounds, not to
#match any data cache; changing them moves the scores, so document any change.
#  wind 7-11 m/s: ~7 m/s hub-height mean is the lower bound of economically
#    viable offshore wind, ~11 m/s is a world-class uk offshore resource, so the
#    band spans marginal to excellent across realistic uk hub-height means.
#  port 0-200 km: ~200 km is a practical ceiling for o&m port accessibility,
#    beyond which routine maintenance logistics become impractical.
SCORE_ANCHORS = {
    "wind_floor_ms":  7.0,    #mean wind at or below this scores 0
    "wind_ceil_ms":  11.0,    #mean wind at or above this scores 1
    "port_max_km":  200.0,    #port distance at or above this scores 0 (0 km scores 1)
}

#economic assumptions
ECONOMICS = {
    "capex_per_mw_fixed_gbp":    3_000_000,   #£3m/mw (higgins & foley baseline)
    "capex_per_mw_floating_gbp": 4_500_000,   #£4.5m/mw floating premium
    "opex_pct_capex_per_year":       0.025,   #2.5% of capex p.a.
    "project_lifetime_years":           25,
    "discount_rate":                  0.07,   #7%
}

#renewables.ninja api
#paste your personal api token below (replacing the placeholder).
#get one free at https://www.renewables.ninja (account -> api).
NINJA_API = {
    "base_url": "https://www.renewables.ninja/api/data/wind",
    "token":    "YOUR_TOKEN_HERE",
    "year":     2019,
    "capacity": 1.0,
    "turbine":  "Vestas V164 8000",
    "height":   120,
    "raw":      True,
}

#era5 reanalysis wind (phase 4)
#when this netcdf exists the pipeline uses real era5 100 m wind for both the
#suitability field and the capacity-factor time series; otherwise it falls back
#to synthetic wind so the pipeline still runs with no download (see
#make_wind_provider). path is relative to the project root.
ERA5 = {
    "nc_path": "data/raw/era5/era5_uk_100m_2019.nc",
    "year":     2019,
}

#turbine power curve for era5-native capacity factors (bug ledger #2)
#official reference turbine: the iea wind 15 mw reference turbine (gaertner et
#al. 2020, nrel/tp-5000-75698; iea-15-240-rwt). it is used in place of the
#vestas v236-15.0 because the v236 curve is not openly published, which would
#break the single-stack reproducibility claim; the iea 15 mw is the same 15 mw
#low-specific-power class and is fully open, so the paper cites this turbine.
#stored as (hub-height wind speed m/s, capacity factor 0-1) points, linearly
#interpolated; cf is forced to 0 at and above cut_out_ms. rated 15 mw is reached
#at ~10.6 m/s (cut-in 3, cut-out 25), which is why a modern low-specific-power
#machine yields genuinely high capacity factors relative to legacy turbines.
POWER_CURVE = {
    "name":       "IEA Wind 15 MW Reference Turbine (Gaertner et al. 2020)",
    "cut_out_ms":  25.0,
    "curve": [
        (0.0, 0.000), (3.0, 0.000), (4.0, 0.052), (5.0, 0.113),
        (6.0, 0.203), (7.0, 0.333), (8.0, 0.508), (9.0, 0.737),
        (10.0, 0.960), (10.6, 1.000), (25.0, 1.000),
    ],
}

#output paths
OUTPUT = {
    "maps_dir":   "outputs/maps",
    "charts_dir": "outputs/charts",
    "data_dir":   "outputs/data",
}