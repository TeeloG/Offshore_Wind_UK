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

#reference wind farm (for shared-infrastructure amortisation and footprint)
#candidate-site economics assume a standardised square array of identical
#turbines. shared transmission (offshore substation / hvdc converter + export
#cable) is amortised across the whole array. array/wake losses stay empirical
#(the loss chain); no wake model is applied here (block 2 phase 2).
#default: 10 x 10 x 15 mw = 1500 mw, replacing the old 500 mw (~33-turbine) ref.
REFERENCE_FARM = {
    "layout_side":         10,    #turbines per side; layout_side^2 turbines total
    "spacing_rotor_diam":   7.0,  #spacing between turbines in rotor diameters
}

#offshore transmission cost assumptions (block 2 phase 2)
#the offshore substation / hvdc converter scale with farm capacity (£/mw) rather
#than a fixed lump, so a larger array is not understated when the shared cost is
#amortised per turbine. the export cable scales with the export-cable length
#(£/km). figures are midpoints of typical ranges in offshore wind cost reviews
#(bvg associates, guide to an offshore wind farm, 2019; ore catapult): hvac
#offshore substation ~£0.1-0.15M/mw, hvdc converter pair ~£0.2-0.4M/mw. these
#assume a roughly linear £/mw scaling; mild economies of scale with capacity
#could refine them later.
TRANSMISSION = {
    "hvdc_threshold_km":            80,        #export distance above which hvdc is used
    "hvac_substation_gbp_per_mw":   120_000,   #£0.12M/mw offshore ac substation
    "hvac_cable_gbp_per_km":      1_200_000,   #ac export cable
    "hvdc_converter_gbp_per_mw":    300_000,   #£0.30M/mw hvdc converter pair
    "hvdc_cable_gbp_per_km":        800_000,   #dc export cable
}

#economic assumptions
ECONOMICS = {
    "capex_per_mw_fixed_gbp":    3_000_000,   #£3m/mw (higgins & foley baseline)
    "capex_per_mw_floating_gbp": 4_500_000,   #£4.5m/mw floating premium
    "opex_pct_capex_per_year":       0.025,   #2.5% of capex p.a.
    "project_lifetime_years":           25,
    "discount_rate":                  0.07,   #7%
}

#monte carlo uncertainty quantification 
#ranges feed src/uncertainty.py: per-site monte carlo LCOE distributions
#(P10/P50/P90) and sobol sensitivity (salib). the central case matches the
#deterministic baseline (discount-rate mode 7%, capex multipliers centred on 1.0).
UQ = {
    "n_samples":  10_000,             #plain monte carlo draws per site
    "sobol_n":     1024,              #saltelli base sample for sobol indices
    "seed":          42,
    #discount rate: triangular (min, mode, max); mode matches ECONOMICS baseline
    "discount_rate_tri": (0.05, 0.07, 0.10),
    #per-component capex multiplier: +/- fraction (triangular about 1.0),
    #differentiated by component maturity (loosely per bvg / industry ranges)
    "capex_pct": {
        "turbine":      0.15,
        "foundation":   0.25,
        "array":        0.20,
        "transmission": 0.30,
        "installation": 0.25,
    },
    #operational losses: normal (mean %, sd percentage-points), clipped >= 0
    "array_loss":      (11.27, 1.5),
    "electrical_loss":  (8.90, 1.5),
    "downtime_loss":    (5.97, 1.5),
    #inter-annual capacity-factor multiplier: normal (mean, sd).
    #TODO: sd 0.05 is a literature stand-in (uk annual wind iav ~5-6%); replace
    #with the measured inter-annual sd once multi-year era5 is available.
    "cf_interannual":   (1.00, 0.05),
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