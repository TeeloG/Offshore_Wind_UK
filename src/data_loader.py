#src/data_loader.py - dataset preparation and api/cache helpers.
#provides placeholder data for development and hooks for real era5/gebco/v2 inputs.

import os
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr


#wind resource provider seam
#a small interface so the pipeline can swap the wind source (synthetic vs era5)
#without touching downstream code. synthetic stays the default so the pipeline
#runs offline/tokenless; era5 (phase 4) activates only when its .nc file exists.

@runtime_checkable
class WindResource(Protocol):
    """
    interface for a wind data source feeding site selection and energy yield.

    implementations provide a spatial mean-wind field (for suitability scoring)
    and an hourly capacity-factor series per point (for aep / dunkelflaute).
    """

    def get_grid(self, resolution: float = 0.5) -> pd.DataFrame:
        """return a grid dataframe with at least lon, lat, mean_wind_ms columns."""
        ...

    def get_hourly_cf(self, lat: float, lon: float, year: int = 2019) -> pd.Series:
        """return an hourly capacity-factor series (0-1) for one point."""
        ...


class SyntheticWind:
    """
    the default wind provider, combining the original synthetic field with the
    renewables.ninja cache/synthetic capacity-factor path.

    keeps the pipeline reproducible with no era5 download and no api token.
    construction params carry the context the underlying helpers need so the
    method signatures stay faithful to the WindResource protocol.

    args:
        eez_shapefile: optional path to the uk eez .shp for grid clipping.
        ninja_token:   renewables.ninja api token ("" gives a synthetic cf series).
        v2_farms_df:   optional v2 farms frame used to seed synthetic cf means.
    """

    def __init__(self, eez_shapefile: str = None,
                 ninja_token: str = "",
                 v2_farms_df=None):
        self.eez_shapefile = eez_shapefile
        self.ninja_token = ninja_token
        self.v2_farms_df = v2_farms_df

    def get_grid(self, resolution: float = 0.5) -> pd.DataFrame:
        return make_placeholder_wind_grid(
            resolution_deg=resolution,
            eez_shapefile=self.eez_shapefile,
        )

    def get_hourly_cf(self, lat: float, lon: float, year: int = 2019) -> pd.Series:
        return fetch_ninja_capacity_factors(
            lat=lat, lon=lon,
            token=self.ninja_token,
            year=year,
            v2_farms_df=self.v2_farms_df,
        )


#era5 wind reanalysis

def load_era5_wind(filepath: str) -> "xr.Dataset":
    """
    load and preprocess era5 wind speed netcdf file.

    expected file: era5 u10/v10 (10 m) or 100 m wind components from cds:
                   https://cds.climate.copernicus.eu
    returns:       xarray dataset with 'u100', 'v100', 'ws100' after
                   log-law extrapolation to hub height.

    todo: replace the stub below with real loading logic once data is ready.
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(filepath)
        print(f"[ERA5] Loaded {filepath} — variables: {list(ds.data_vars)}")
        return ds
    except FileNotFoundError:
        print(f"[ERA5] WARNING: File not found: {filepath}. Returning None.")
        return None


def make_placeholder_wind_grid(resolution_deg: float = 0.5,
                                eez_shapefile: str = None) -> pd.DataFrame:
    """
    generate a synthetic wind speed grid over uk waters for development use.
    delete this once real era5 data is available.

    if eez_shapefile is provided, grid points outside the uk eez are removed.
    download eez shapefile from https://www.marineregions.org (search for
    "united kingdom exclusive economic zone") and place in data/raw/eez/.

    args:
        resolution_deg: grid spacing in degrees (default 0.5 ≈ 55 km).
        eez_shapefile:  path to eez .shp file, e.g. "data/raw/eez/eez.shp".

    returns:
        dataframe with columns: lon, lat, mean_wind_ms, wind_power_density
    """
    from config import UK_BBOX

    lons = np.arange(UK_BBOX["min_lon"], UK_BBOX["max_lon"], resolution_deg)
    lats = np.arange(UK_BBOX["min_lat"], UK_BBOX["max_lat"], resolution_deg)

    rows = []
    rng = np.random.default_rng(42)

    for lat in lats:
        for lon in lons:
            base = 8.5 + (lat - 49.5) * 0.15 + abs(lon) * 0.1
            ws = max(0, rng.normal(base, 0.8))
            wpd = 0.5 * 1.225 * ws**3
            rows.append({
                "lon": lon,
                "lat": lat,
                "mean_wind_ms":       round(ws, 2),
                "wind_power_density": round(wpd, 1),
            })

    df = pd.DataFrame(rows)
    print(f"[Placeholder] Generated {len(df)} synthetic wind grid points.")

    #eez clipping
    if eez_shapefile is not None:
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            eez = gpd.read_file(eez_shapefile).to_crs(epsg=4326)

            gdf = gpd.GeoDataFrame(
                df,
                geometry=[Point(lon, lat) for lon, lat in zip(df.lon, df.lat)],
                crs="EPSG:4326"
            )

            df = (gpd.sjoin(gdf, eez, how="inner", predicate="within")
                     .drop(columns=["geometry", "index_right"])
                     .reset_index(drop=True))

            print(f"[Placeholder] {len(df)} points remaining after EEZ mask.")

        except Exception as e:
            print(f"[EEZ] WARNING: Could not apply EEZ mask: {e}. Using full bounding box.")

    return df


#gebco bathymetry

def add_gebco_depth(grid_df: pd.DataFrame,
                    gebco_nc_path: str) -> pd.DataFrame:
    """
    add a 'depth_m' column to the wind grid using gebco 2023 bathymetry data.

    gebco stores elevation relative to sea level: negative = ocean, positive = land.
    we take the absolute value of negative cells to express depth as a positive number.

    download gebco netcdf from https://download.gebco.net:
        - uk bounding box (~-16 to 4 lon, 47 to 65 lat)
        - 15 arc-second grid resolution (~450 m at uk latitudes)
        - format: netcdf

    args:
        grid_df:       dataframe with 'lat' and 'lon' columns.
        gebco_nc_path: path to gebco netcdf, e.g. "data/raw/gebco/gebco_2023_uk.nc"

    returns:
        grid_df with 'depth_m' column added. land/ambiguous points get depth_m = 0.
    """
    try:
        import xarray as xr

        print(f"[GEBCO] Loading bathymetry from {gebco_nc_path} ...")
        ds = xr.open_dataset(gebco_nc_path)

        #handle both 'lon'/'lat' and 'longitude'/'latitude' coordinate naming conventions
        elev_var = "elevation" if "elevation" in ds else list(ds.data_vars)[0]
        lon_dim  = "lon"  if "lon"  in ds.coords else "longitude"
        lat_dim  = "lat"  if "lat"  in ds.coords else "latitude"

        df = grid_df.copy()
        depths = []

        for _, row in df.iterrows():
            elev = float(
                ds[elev_var].sel(
                    {lon_dim: row["lon"], lat_dim: row["lat"]},
                    method="nearest"  #nearest-neighbour lookup
                )
            )
            depths.append(abs(elev) if elev < 0 else 0.0)

        df["depth_m"] = depths

        n_ocean = (df["depth_m"] > 0).sum()
        print(f"[GEBCO] Depth data added. {n_ocean}/{len(df)} points have ocean depth > 0 m. "
              f"Range: {df['depth_m'].min():.0f}–{df['depth_m'].max():.0f} m")
        return df

    except FileNotFoundError:
        print(f"[GEBCO] WARNING: File not found at {gebco_nc_path}. "
              f"Depth column will not be added — economics module will fall back to 40 m placeholder.")
        return grid_df

    except Exception as e:
        print(f"[GEBCO] WARNING: Could not load bathymetry: {e}. "
              f"Depth column will not be added.")
        return grid_df


#port distance

def add_port_distances(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    add a 'dist_to_port_km' column to the wind grid.

    computes straight-line distance from each grid point to the nearest major
    uk offshore wind o&m or construction port using a vectorised haversine matrix.

    port list from crown estate offshore wind report 2023 and ore catapult uk
    port infrastructure survey 2022. only ports with significant offshore wind
    activity are included.

    args:
        grid_df: dataframe with 'lat' and 'lon' columns.

    returns:
        grid_df with 'dist_to_port_km' and 'nearest_port' columns added.
    """
    try:
        UK_PORTS = [
            #north sea - east england
            ("Hull",              53.744,  -0.332),
            ("Grimsby/Immingham", 53.574,  -0.080),
            ("Great Yarmouth",    52.608,   1.729),
            ("Lowestoft",         52.469,   1.751),
            ("Blyth",             55.127,  -1.503),
            #north sea - east scotland
            ("Aberdeen",          57.144,  -2.096),
            ("Peterhead",         57.502,  -1.782),
            ("Methil",            56.173,  -3.007),
            ("Leith (Edinburgh)", 55.975,  -3.173),
            #irish sea
            ("Barrow-in-Furness", 54.111,  -3.227),
            ("Belfast",           54.607,  -5.930),
            ("Mostyn",            53.319,  -3.263),
            #northern scotland / atlantic
            ("Wick",              58.441,  -3.089),
            ("Nigg",              57.691,  -4.038),
            #south
            ("Harwich",           51.944,   1.285),
            ("Newhaven",          50.793,   0.055),
        ]

        #vectorised haversine rather than a degree-based kd-tree: at 58-60°n one
        #degree of longitude is only ~57 km east-west vs 111 km north-south,
        #which caused the previous approach to assign wrong nearest ports.
        def haversine_matrix(lats1, lons1, lats2, lons2):
            """returns full (n_grid × n_ports) distance matrix in km."""
            R = 6371.0
            lat1 = np.radians(lats1)[:, None]
            lat2 = np.radians(lats2)[None, :]
            lon1 = np.radians(lons1)[:, None]
            lon2 = np.radians(lons2)[None, :]
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            return R * 2 * np.arcsin(np.sqrt(a))

        port_lats = np.array([lat for _, lat, lon in UK_PORTS])
        port_lons = np.array([lon for _, lat, lon in UK_PORTS])
        grid_lats = grid_df["lat"].values
        grid_lons = grid_df["lon"].values

        dist_matrix = haversine_matrix(grid_lats, grid_lons, port_lats, port_lons)

        nearest_idx = dist_matrix.argmin(axis=1)
        dists_km    = dist_matrix.min(axis=1)

        df = grid_df.copy()
        df["dist_to_port_km"] = np.round(dists_km, 1)
        df["nearest_port"]    = [UK_PORTS[i][0] for i in nearest_idx]

        print(f"[Ports] Haversine distance to nearest port added. "
              f"Range: {dists_km.min():.0f}–{dists_km.max():.0f} km. "
              f"Mean: {dists_km.mean():.0f} km.")
        return df

    except ImportError:
        print("[Ports] WARNING: scipy not installed. "
              "Run 'pip install scipy' to enable port distance calculation. "
              "Suitability scoring will use placeholder 0.5 for distance_to_port.")
        return grid_df

    except Exception as e:
        print(f"[Ports] WARNING: Could not compute port distances: {e}. "
              "Suitability scoring will use placeholder 0.5 for distance_to_port.")
        return grid_df


#marine protected areas

def load_marine_protected_areas(shapefile_path: str) -> "gpd.GeoDataFrame":
    """
    load uk marine protected areas from jncc / natural england shapefile.

    source: https://jncc.gov.uk/our-work/uk-marine-protected-area-datasets-for-download/
    download, unzip, and place .shp/.dbf/.shx/.prj files in data/raw/mpa/

    returns: geodataframe in epsg:4326 with mpa polygons (hard exclusion zones).
    """
    try:
        import geopandas as gpd

        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

        name_cols = ["name", "Name", "OBJECTID", "AREA_NAME", "CATEGORY"]
        name_col = next((col for col in name_cols if col in gdf.columns), None)

        if name_col:
            print(f"[MPA] Loaded {len(gdf)} protected area polygons from {os.path.basename(shapefile_path)}.")
            print(f"[MPA] Categories: {gdf[name_col].unique()[:5]}...")
        else:
            print(f"[MPA] Loaded {len(gdf)} protected area polygons.")

        return gdf

    except FileNotFoundError:
        print(f"[MPA] WARNING: File not found at {shapefile_path}.")
        print(f"[MPA]          Download JNCC MPAs from: https://jncc.gov.uk/our-work/uk-marine-protected-area-datasets-for-download/")
        print(f"[MPA]          Extract and place in: data/raw/mpa/")
        return None

    except Exception as e:
        print(f"[MPA] WARNING: Could not load shapefile: {e}. Returning None.")
        return None


#bgs seabed substrate

def load_seabed_substrate(shapefile_path: str) -> "gpd.GeoDataFrame":
    """
    load bgs 250k seabed sediments shapefile.

    source: bgs digimap - offshore-250k-seabedsed
    file:   bgs_250k_seabedsediments_wgs84_v3.shp

    extracts folk_d50 (plain-english folk classification e.g. 'sand', 'gravel',
    'mud') and renames it to 'seabed_type' for use in score_soft_constraints().

    returns: geodataframe with 'seabed_type' and 'geometry' columns.
    """
    try:
        import geopandas as gpd

        gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

        #rename folk_d50 so score_soft_constraints can find it by name
        gdf = gdf[["FOLK_D50", "geometry"]].copy()
        gdf = gdf.rename(columns={"FOLK_D50": "seabed_type"})

        #lowercase for case-insensitive mapping in suitability.py
        gdf["seabed_type"] = gdf["seabed_type"].str.lower().str.strip()

        print(f"[Seabed] Loaded {len(gdf)} BGS sediment polygons.")
        print(f"[Seabed] Types present: {sorted(gdf['seabed_type'].dropna().unique())}")
        return gdf

    except Exception as e:
        print(f"[Seabed] WARNING: Could not load shapefile: {e}. Returning None.")
        return None


#v2 wind farm data

def load_v2_wind_farms(filepath: str) -> pd.DataFrame:
    """
    load real uk offshore wind farm data from the v2 project excel file.

    parses sheet 3 which contains renewables.ninja capacity factors, loss factors,
    and coordinates for 44 operational uk wind farms.

    args:
        filepath: path to the v2 excel file.

    returns:
        dataframe with columns: name, lat, lon, ninja_cf, calculated_cf.
        returns none if the file cannot be loaded.
    """
    import re

    try:
        df = pd.read_excel(filepath, sheet_name="Sheet3")
        df = df.dropna(subset=["Name"])  #remove blank rows at end of sheet

        lats, lons = [], []
        for s in df["Latitude & Longitude"]:
            s = str(s).strip().replace("\n", " ")
            nums = re.findall(r"-?\d+\.?\d*", s)
            lat = float(nums[0])
            lon = float(nums[1])
            if "S" in s: lat = -lat
            if "W" in s and lon > 0: lon = -lon
            lats.append(lat)
            lons.append(lon)

        result = pd.DataFrame({
            "name":           df["Name"].values,
            "lat":            lats,
            "lon":            lons,
            "location":       df["Location"].values,
            "turbine_type":   df["Turbine Type"].values,
            "ninja_cf":       df["Renewables Ninja Capacity Factor (%)"].values / 100,
            "calculated_cf":  df["Calculated overall Capacity Factor"].values / 100,
        })

        #pull capacity and fixed/floating classification from sheet2
        df2 = pd.read_excel(filepath, sheet_name="Sheet2")[
            ["Name", "Fixed/Floating", "Capacity (MW)"]
        ]
        result = result.merge(df2.rename(columns={
            "Name":           "name",
            "Fixed/Floating": "fixed_floating",
            "Capacity (MW)":  "capacity_mw",
        }), on="name", how="left")

        print(f"[V2] Loaded {len(result)} wind farms. "
              f"CF range: {result['ninja_cf'].min():.1%}–{result['ninja_cf'].max():.1%}")
        return result

    except Exception as e:
        print(f"[V2] WARNING: Could not load V2 data: {e}. Will use synthetic fallback.")
        return None


def load_v2_default_losses(filepath: str) -> dict:
    """
    extract default loss factors from the bottom row of sheet3 in the v2 excel file.

    the last row contains average loss percentages used as defaults for candidate sites.

    args:
        filepath: path to the v2 excel file.

    returns:
        dict with keys: array_loss_pct, electrical_loss_pct, downtime_pct (0-100 range).
        falls back to hardcoded defaults if the file cannot be loaded.
    """
    try:
        df = pd.read_excel(filepath, sheet_name="Sheet3")
        last_row = df.iloc[-1]

        if pd.isna(last_row.get("Name")):
            losses = {
                "array_loss_pct":       float(last_row.get("Array Losses (%)", 11.27)),
                "electrical_loss_pct":  float(last_row.get("Electrical Losses (%)", 8.90)),
                "downtime_pct":         float(last_row.get("Downtime (%)", 5.97)),
            }
            print(f"[V2-Losses] Default loss factors loaded: "
                  f"{losses['array_loss_pct']:.2f}% array, "
                  f"{losses['electrical_loss_pct']:.2f}% electrical, "
                  f"{losses['downtime_pct']:.2f}% downtime")
            return losses
        else:
            print("[V2-Losses] WARNING: Last row of Sheet3 is not a defaults row. "
                  "Using fallback default loss values.")
            return {
                "array_loss_pct": 11.27,
                "electrical_loss_pct": 8.90,
                "downtime_pct": 5.97,
            }

    except Exception as e:
        print(f"[V2-Losses] WARNING: Could not extract loss factors: {e}. "
              "Using fallback default values.")
        return {
            "array_loss_pct": 11.27,
            "electrical_loss_pct": 8.90,
            "downtime_pct": 5.97,
        }


#renewables.ninja capacity factors

def fetch_ninja_capacity_factors(lat: float, lon: float, token: str,
                                  year: int = 2019,
                                  v2_farms_df=None) -> pd.Series:
    """
    retrieve hourly capacity factor time series from the renewables.ninja api.

    docs:    https://www.renewables.ninja/documentation
    returns: pd.series indexed by datetime, values 0-1.

    responses are cached locally as csv to avoid hitting rate limits on repeat runs.
    if no token is provided, falls back to synthetic data using v2 farm cfs as a mean.
    """
    import requests
    from config import NINJA_API

    cache_file = f"outputs/data/ninja_{lat}_{lon}_{year}.csv"

    if os.path.exists(cache_file):
        print(f"[Ninja] Cache hit: {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)["electricity"]

    if not token:
        idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
        seed = int(abs(lat * 1000 + lon * 100)) % (2**31)

        #if v2 data is available, use the nearest real farm's cf as the synthetic mean;
        #otherwise fall back to a simple latitude-based approximation.
        if v2_farms_df is not None:
            from scipy.spatial import cKDTree
            coords = v2_farms_df[["lat", "lon"]].values
            tree = cKDTree(coords)
            dist, idx_nearest = tree.query([[lat, lon]])
            mean_cf = float(v2_farms_df.iloc[idx_nearest[0]]["ninja_cf"])
            nearest_name = v2_farms_df.iloc[idx_nearest[0]]["name"]
            print(f"[Ninja] Using V2 CF from nearest farm: "
                  f"{nearest_name} ({mean_cf:.1%})")
        else:
            mean_cf = 0.35 + (lat - 50.0) * (0.15 / 13.0)
            mean_cf = float(np.clip(mean_cf, 0.30, 0.55))
            print(f"[Ninja] No API token or V2 data — latitude-based CF: {mean_cf:.1%}")

        return pd.Series(
            np.clip(np.random.default_rng(seed).normal(mean_cf, 0.15, 8760), 0, 1),
            index=idx,
            name="electricity"
        )

    #live api call
    headers = {"Authorization": f"Token {token}"}

    params = {
        "lat":       lat,
        "lon":       lon,
        "date_from": f"{year}-01-01",
        "date_to":   f"{year}-12-31",
        "capacity":  NINJA_API["capacity"],
        "height":    NINJA_API["height"],
        "turbine":   NINJA_API["turbine"],
        "format":    "json",
        "raw":       str(NINJA_API["raw"]).lower(),
    }

    resp = requests.get(NINJA_API["base_url"], headers=headers, params=params, timeout=30)
    resp.raise_for_status()

    df = pd.DataFrame.from_dict(resp.json()["data"], orient="index")
    df.index = pd.to_datetime(pd.to_numeric(df.index), unit='ms')

    os.makedirs("outputs/data", exist_ok=True)
    df.to_csv(cache_file)

    print(f"[Ninja] Fetched and cached {len(df)} hourly values for ({lat}, {lon}).")
    return df["electricity"]