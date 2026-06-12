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

#soft constraint weights
#must sum to 1.0 so the composite score stays on a 0-1 scale.
SOFT_WEIGHTS = {
    "wind_resource":    0.40,
    "water_depth":      0.25,
    "distance_to_port": 0.20,
    "seabed_type":      0.15,
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

#output paths
OUTPUT = {
    "maps_dir":   "outputs/maps",
    "charts_dir": "outputs/charts",
    "data_dir":   "outputs/data",
}