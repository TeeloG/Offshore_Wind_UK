#src/visualisation.py - interactive maps (folium) and charts (plotly).
#
#folium produces interactive html maps (leaflet.js) that open in any browser
#without a server. plotly produces interactive html charts with hover and zoom.
#static png exports (make_static_*) use geopandas + matplotlib for the dissertation pdf.

import json
import os

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import UK_BBOX, REGIONS_OF_INTEREST, OUTPUT


#helpers

def _ensure_output_dirs():
    """create all output directories defined in config.output if they don't exist."""
    for path in OUTPUT.values():
        os.makedirs(path, exist_ok=True)


def _add_eez_overlay(m: folium.Map, eez_shapefile: str | None) -> None:
    """
    draw the eez boundary on the map and zoom to it.

    folium maps are always rectangular; this makes the data and view follow the eez
    rather than uk_bbox. without this, clipped points can still look 'rectangular'
    until you also clip the grid in data_loader and pass the same path from main.
    """
    if not eez_shapefile or not os.path.isfile(eez_shapefile):
        return
    try:
        import geopandas as gpd

        eez = gpd.read_file(eez_shapefile).to_crs(epsg=4326)
        if eez.empty:
            return

        fg = folium.FeatureGroup(name="UK EEZ boundary", show=True)
        geo = json.loads(eez.to_json())
        folium.GeoJson(
            geo,
            style_function=lambda _feat: {
                "fillColor": "#1e90ff",
                "color": "#0d47a1",
                "weight": 2,
                "fillOpacity": 0.06,
            },
        ).add_to(fg)
        fg.add_to(m)

        minx, miny, maxx, maxy = eez.total_bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])
    except Exception as exc:
        print(f"[Vis] EEZ boundary overlay skipped: {exc}")


def _wind_speed_to_color(ws: float) -> str:
    """
    map a wind speed (m/s) to a hex colour for map markers.
    blue - green - yellow - red with increasing wind resource.
    """
    if   ws < 7:   return "#3182bd"   #blue  - poor
    elif ws < 9:   return "#41ab5d"   #green - moderate
    elif ws < 11:  return "#f6d543"   #yellow - good
    else:          return "#e31a1c"   #red   - excellent


#map 1: wind resource

def make_wind_resource_map(wind_grid_df: pd.DataFrame,
                            filename: str = "wind_resource_map.html",
                            eez_shapefile: str | None = None) -> str:
    """
    create an interactive folium map showing mean wind speed across uk waters.

    layers:
        - heatmap of wind power density
        - circlemarkers for wind speed at each grid point
        - popup with lat/lon/wind speed on click
        - markers for regions of interest

    args:
        wind_grid_df: dataframe with lon, lat, mean_wind_ms, wind_power_density
        filename:     output html filename (saved to outputs/maps/)

    returns:
        absolute path of the saved html file.
    """
    _ensure_output_dirs()

    centre_lat = (UK_BBOX["min_lat"] + UK_BBOX["max_lat"]) / 2
    centre_lon = (UK_BBOX["min_lon"] + UK_BBOX["max_lon"]) / 2

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    _add_eez_overlay(m, eez_shapefile)

    #heatmap layer
    heat_data = [
        [row["lat"], row["lon"], row["wind_power_density"]]
        for _, row in wind_grid_df.iterrows()
    ]

    HeatMap(
        heat_data,
        name="Wind Power Density (W/m²)",
        min_opacity=0.3,
        max_zoom=10,
        radius=18,
        blur=12,
    ).add_to(m)

    #wind speed markers
    wind_fg = folium.FeatureGroup(name="Wind Speed Grid Points", show=False)

    for _, row in wind_grid_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=_wind_speed_to_color(row["mean_wind_ms"]),
            fill=True,
            fill_opacity=0.7,
            weight=0.5,
            popup=folium.Popup(
                f"<b>Wind Resource</b><br>"
                f"Lat: {row['lat']:.2f}°, Lon: {row['lon']:.2f}°<br>"
                f"Mean Wind Speed: {row['mean_wind_ms']:.1f} m/s<br>"
                f"Wind Power Density: {row['wind_power_density']:.0f} W/m²",
                max_width=200
            ),
        ).add_to(wind_fg)

    wind_fg.add_to(m)

    #regions of interest
    roi_fg = folium.FeatureGroup(name="Regions of Interest")

    for name, (lon, lat) in REGIONS_OF_INTEREST.items():
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color="darkblue", icon="cloud", prefix="fa"),
            tooltip=name,
        ).add_to(roi_fg)

    roi_fg.add_to(m)

    #legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);
                font-family:sans-serif;font-size:12px;">
      <b>Mean Wind Speed (m/s)</b><br>
      <span style="color:#3182bd">■</span> &lt; 7 m/s<br>
      <span style="color:#41ab5d">■</span> 7–9 m/s<br>
      <span style="color:#f6d543">■</span> 9–11 m/s<br>
      <span style="color:#e31a1c">■</span> &gt; 11 m/s
    </div>"""

    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = os.path.join(OUTPUT["maps_dir"], filename)
    m.save(out_path)
    print(f"[Vis] Wind resource map saved -> {out_path}")
    return os.path.abspath(out_path)


#map 2: site suitability

def make_suitability_map(scored_df: pd.DataFrame,
                          candidate_sites: pd.DataFrame = None,
                          filename: str = "suitability_map.html",
                          eez_shapefile: str | None = None,
                          v2_farms_df: pd.DataFrame = None,
                          site_combinations: dict = None,
                          sites_final_df: pd.DataFrame = None) -> str:
    """
    creates an interactive suitability map with:
        - colour-coded grid points by suitability score
        - hard exclusion zones highlighted in red
        - candidate site markers (if provided)
        - real uk wind farm markers from v2 data (if provided)
        - interactive parameter toggle panel (if site_combinations provided)

    site_combinations: dict from precompute_all_combinations() in the form
        {'mpa_excluded': {...}, 'mpa_allowed': {...}}. when provided, the static
        candidate_sites markers are replaced by a js-driven toggle panel embedded
        directly on the map - no separate window or page reload needed.

    sites_final_df: fully enriched candidate sites dataframe from phase 4
        (mean_cf, aep_adjusted_mwh, lcoe_gbp_per_mwh, capex_gbp_millions).
        used to populate popup economic fields for sites that appear in the
        full-parameter run; other combinations show dashes for missing fields.
    """
    _ensure_output_dirs()

    centre_lat = (UK_BBOX["min_lat"] + UK_BBOX["max_lat"]) / 2
    centre_lon = (UK_BBOX["min_lon"] + UK_BBOX["max_lon"]) / 2

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=6,
        tiles="CartoDB positron",
        prefer_canvas=True
    )

    _add_eez_overlay(m, eez_shapefile)

    def score_color(score):
        """convert a 0-1 suitability score to a traffic-light hex colour."""
        if pd.isna(score): return "#cccccc"
        if score > 0.75:   return "#1a9641"
        if score > 0.50:   return "#a6d96a"
        if score > 0.25:   return "#fdae61"
        return "#d7191c"

    suit_fg = folium.FeatureGroup(name="Suitability Score")
    excl_fg = folium.FeatureGroup(name="Hard Exclusions", show=True)

    for _, row in scored_df.iterrows():
        lat, lon = row["lat"], row["lon"]

        if row.get("hard_excluded", False):
            reason = row.get("exclusion_reason", "Hard excluded")
            depth  = row.get("depth_m", None)
            depth_str = f"<br>Depth: {depth:.0f} m" if depth is not None else ""

            folium.CircleMarker(
                [lat, lon],
                radius=4,
                color="#d7191c",
                fill=True,
                fill_opacity=0.4,
                weight=0,
                tooltip="Hard excluded",
                popup=folium.Popup(
                    f"<b>Hard Excluded</b>{depth_str}<br>"
                    f"<b>Reason:</b> {reason}",
                    max_width=240
                ),
            ).add_to(excl_fg)

        else:
            score = row.get("suitability_score", np.nan)
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                color=score_color(score),
                fill=True,
                fill_opacity=0.7,
                weight=0,
                popup=folium.Popup(
                    f"<b>Suitability: {score:.3f}</b><br>"
                    f"Wind score: {row.get('score_wind', '–'):.2f} "
                    f"({row.get('mean_wind_ms', '–'):.1f} m/s)<br>"
                    f"Depth score: {row.get('score_depth', '–'):.2f} "
                    f"({row.get('depth_m', '–'):.0f} m)<br>"
                    f"Port score: {row.get('score_port', '–'):.2f} "
                    f"({row.get('dist_to_port_km', '–'):.0f} km to {row.get('nearest_port', '–')})<br>"
                    f"Seabed score: {row.get('score_seabed', '–'):.2f} "
                    f"({row.get('seabed_type', 'Unknown')})",
                    max_width=300
                ),
            ).add_to(suit_fg)

    suit_fg.add_to(m)
    excl_fg.add_to(m)

    #candidate site markers
    #when site_combinations is provided, the toggle panel injects markers via js.
    #otherwise fall back to static featuregroup markers.
    if site_combinations is not None:
        #all parameter combinations were pre-computed in python and are embedded
        #as a single json blob. js reads the active checkboxes, looks up the
        #matching combination, and redraws numbered markers client-side.

        map_var   = m.get_name()   #e.g. "map_a3f1c2" - folium's js variable name
        site_json = json.dumps(site_combinations)

        #economic lookup keyed by "lat,lon" (2 dp) so the js popup can show
        #cf/aep/lcoe for sites that went through phase 4.
        econ_lookup = {}
        if sites_final_df is not None:
            for _, row in sites_final_df.iterrows():
                key = f"{row['lat']:.2f},{row['lon']:.2f}"
                econ_lookup[key] = {
                    "mean_cf":       round(float(row["mean_cf"]), 4)            if pd.notna(row.get("mean_cf"))               else None,
                    "calculated_cf": round(float(row["calculated_cf"]), 4)      if pd.notna(row.get("calculated_cf"))         else None,
                    "aep":           round(float(row["aep_adjusted_mwh"]), 0)   if pd.notna(row.get("aep_adjusted_mwh"))      else None,
                    "capex":         round(float(row["capex_gbp_millions"]), 1)  if pd.notna(row.get("capex_gbp_millions"))    else None,
                    "lcoe":          round(float(row["lcoe_gbp_per_mwh"]), 1)   if pd.notna(row.get("lcoe_gbp_per_mwh"))      else None,
                }
        econ_json = json.dumps(econ_lookup)

        toggle_html = f"""
<div id="param-toggle-panel" style="
    position:fixed;bottom:30px;right:30px;z-index:9999;
    background:white;padding:14px 16px;border-radius:10px;
    box-shadow:0 2px 10px rgba(0,0,0,.22);
    font-family:'Segoe UI',Arial,sans-serif;font-size:12px;
    min-width:210px;line-height:1.7;">
  <div style="font-size:13px;font-weight:700;margin-bottom:2px;">
    Suitability Parameters
  </div>
  <div style="color:#666;font-size:11px;margin-bottom:8px;">
    Toggle to rerank top 15 candidate sites
  </div>
  <label style="display:block;cursor:pointer;">
    <input type="checkbox" class="param-cb" value="wind_resource" checked>
    Wind Resource <span style="color:#999">(40%)</span>
  </label>
  <label style="display:block;cursor:pointer;">
    <input type="checkbox" class="param-cb" value="water_depth" checked>
    Water Depth <span style="color:#999">(25%)</span>
  </label>
  <label style="display:block;cursor:pointer;">
    <input type="checkbox" class="param-cb" value="distance_to_port" checked>
    Port Distance <span style="color:#999">(20%)</span>
  </label>
  <label style="display:block;cursor:pointer;">
    <input type="checkbox" class="param-cb" value="seabed_type" checked>
    Seabed Type <span style="color:#999">(15%)</span>
  </label>
  <label style="display:block;cursor:pointer;margin-top:8px;">
    <input type="checkbox" id="exclude-mpa-cb" checked>
    Exclude Marine Protected Areas
  </label>
  <div id="toggle-status" style="
      margin-top:8px;padding-top:8px;border-top:1px solid #eee;
      color:#555;font-size:11px;line-height:1.5;">
    Loading…
  </div>
</div>

<script>
(function() {{
  var siteData  = {site_json};
  var econData  = {econ_json};

  var BASE_WEIGHTS = {{
    'wind_resource': 0.40, 'water_depth': 0.25,
    'distance_to_port': 0.20, 'seabed_type': 0.15
  }};
  var PARAM_LABELS = {{
    'wind_resource': 'Wind', 'water_depth': 'Depth',
    'distance_to_port': 'Port', 'seabed_type': 'Seabed'
  }};

  var candidateMarkers = [];

  function getMap() {{ return window['{map_var}']; }}

  function fmt(val, prefix, suffix, dp) {{
    if (val === null || val === undefined) return '\u2014';
    return prefix + val.toFixed(dp) + suffix;
  }}

  function updateCandidateMarkers() {{
    var mapObj = getMap();
    if (!mapObj) return;

    candidateMarkers.forEach(function(m) {{ mapObj.removeLayer(m); }});
    candidateMarkers = [];

    var active = [];
    document.querySelectorAll('.param-cb:checked').forEach(function(cb) {{
      active.push(cb.value);
    }});
    active.sort();

    var statusEl = document.getElementById('toggle-status');
    if (active.length === 0) {{
      statusEl.innerHTML = '<span style="color:#c00;">Select at least one parameter.</span>';
      return;
    }}

    var key   = active.join(',');
    var mpaKey = document.getElementById('exclude-mpa-cb').checked
                 ? 'mpa_excluded' : 'mpa_allowed';
    var sites = ((siteData[mpaKey] || {{}})[key]) || [];

    var totalW = active.reduce(function(s, p) {{ return s + (BASE_WEIGHTS[p] || 0); }}, 0);
    var weightDesc = active.map(function(p) {{
      return PARAM_LABELS[p] + '\u202f' + Math.round((BASE_WEIGHTS[p] / totalW) * 100) + '%';
    }}).join(' &middot; ');

    statusEl.innerHTML =
      '<b>' + sites.length + ' sites selected</b><br>' +
      '<span style="color:#777;">' + weightDesc + '</span><br>' +
      '<span style="color:#555;">MPA exclusions: ' +
      (mpaKey === 'mpa_excluded' ? 'enabled' : 'disabled') + '</span>';

    sites.forEach(function(site) {{
      var icon = L.divIcon({{
        html: '<div style="background:#16a34a;color:#fff;border-radius:50%;' +
              'width:26px;height:26px;display:flex;align-items:center;' +
              'justify-content:center;font-weight:700;font-size:11px;' +
              'box-shadow:0 1px 5px rgba(0,0,0,.45);border:2px solid #fff;">' +
              site.rank + '</div>',
        iconSize: [26,26], iconAnchor: [13,13], className: ''
      }});

      // Look up Phase 4 economic data for this lat/lon if available
      var econKey = site.lat.toFixed(2) + ',' + site.lon.toFixed(2);
      var econ    = econData[econKey] || {{}};

      var depthStr = site.depth ? site.depth + '\u202fm' : '\u2014';
      var cfStr    = econ.mean_cf  !== undefined && econ.mean_cf  !== null
                     ? (econ.mean_cf * 100).toFixed(1) + '%' : '\u2014';
      var calcCfStr = econ.calculated_cf  !== undefined && econ.calculated_cf  !== null
                     ? (econ.calculated_cf * 100).toFixed(1) + '%' : '\u2014';
      var aepStr   = econ.aep     !== undefined && econ.aep     !== null
                     ? econ.aep.toLocaleString() + ' MWh' : '\u2014';
      var capexStr = econ.capex   !== undefined && econ.capex   !== null
                     ? '\u00a3' + econ.capex.toFixed(1) + 'M' : '\u2014';
      var lcoeStr  = econ.lcoe    !== undefined && econ.lcoe    !== null
                     ? '\u00a3' + econ.lcoe.toFixed(1) + '/MWh' : '\u2014';

      var popupHtml =
        "<span style='color:green;text-decoration:underline;'>" +
          "<b>Candidate Site " + site.rank + "</b>" +
        "</span><br>" +
        "<i>Proposed location</i><br>" +
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500<br>" +
        "<b>Foundation:</b> " + (site.foundation ? site.foundation.charAt(0).toUpperCase() + site.foundation.slice(1) : '\u2014') + "<br>" +
        "<b>Capacity:</b> 15 MW (per turbine)<br>" +
        "<b>Water depth:</b> " + depthStr + "<br>" +
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500<br>" +
        "<b>Mean CF:</b> " + cfStr + "<br>" +
        "<b>Estimated overall CF:</b> " + calcCfStr + "<br>" +
        "<small style='color:#555;'>(Array, Electrical &amp; Downtime)</small><br>" +
        "<b>AEP:</b> " + aepStr + "<br>" +
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500<br>" +
        "<b>Modelled CapEx:</b> " + capexStr + "<br>" +
        "<b>Modelled LCOE:</b> " + lcoeStr;

      var marker = L.marker([site.lat, site.lon], {{ icon: icon }})
        .addTo(mapObj)
        .bindPopup(popupHtml, {{ maxWidth: 260 }})
        .bindTooltip('Candidate Site #' + site.rank, {{ permanent: false }});

      candidateMarkers.push(marker);
    }});
  }}

  document.querySelectorAll('.param-cb, #exclude-mpa-cb').forEach(function(cb) {{
    cb.addEventListener('change', updateCandidateMarkers);
  }});

  setTimeout(updateCandidateMarkers, 600);
}})();
</script>
"""
        m.get_root().html.add_child(folium.Element(toggle_html))

    elif candidate_sites is not None and len(candidate_sites):
        #static candidate site markers (no toggle panel)
        cand_fg = folium.FeatureGroup(name="Candidate Sites")

        for idx, site in candidate_sites.iterrows():
            mean_cf = site.get('mean_cf', None)
            calc_cf = site.get('calculated_cf', None)

            mean_cf_str = f"{mean_cf:.1%}" if mean_cf is not None and pd.notna(mean_cf) else "–"
            calc_cf_str = f"{calc_cf:.1%}" if calc_cf is not None and pd.notna(calc_cf) else "–"

            folium.Marker(
                location=[site["lat"], site["lon"]],
                tooltip=f"Site {idx} — Score: {site.get('suitability_score', '?'):.3f}",
                popup=folium.Popup(
                    f"<span style='color: green; text-decoration: underline;'><b>Candidate Site {idx}</b></span><br>"
                    f"<i>Proposed location</i><br>"
                    f"──────────────────<br>"
                    f"<b>Foundation:</b> {site.get('foundation_type', '–').title()}<br>"
                    f"<b>Capacity:</b> 15 MW (per turbine)<br>"
                    f"<b>Water depth:</b> {site.get('depth_m', '–'):.0f} m<br>"
                    f"──────────────────<br>"
                    f"<b>Mean CF:</b> {mean_cf_str}<br>"
                    f"<b>Estimated overall CF:</b> {calc_cf_str}<br>"
                    f"<small style='color:#555'>(After Array, Electrical &amp; Downtime Losses)</small><br>"
                    f"<b>AEP:</b> {site.get('aep_adjusted_mwh', '–'):,.0f} MWh<br>"
                    f"──────────────────<br>"
                    f"<b>Modelled CapEx:</b> £{site.get('capex_gbp_millions', '–'):.1f}M<br>"
                    f"<b>Modelled LCOE:</b> £{site.get('lcoe_gbp_per_mwh', '–'):.1f}/MWh",
                    max_width=260
                ),
                icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
            ).add_to(cand_fg)

        cand_fg.add_to(m)

    #real uk wind farm markers (v2 data)
    if v2_farms_df is not None and len(v2_farms_df):
        farms_fg = folium.FeatureGroup(name="Real UK Wind Farms (V2 data)")

        for _, farm in v2_farms_df.iterrows():
            ninja_cf       = farm.get("ninja_cf", None)
            calc_cf        = farm.get("calculated_cf", None)
            turbine        = farm.get("turbine_type", "N/A")
            location       = farm.get("location", "N/A")
            fixed_flt      = farm.get("fixed_floating", "N/A")
            depth          = farm.get("depth_m", None)
            capacity       = farm.get("capacity_mw", None)
            lcoe_turbine   = farm.get("lcoe_gbp_per_mwh", None)
            lcoe_flat      = farm.get("lcoe_v2_flat_gbp_per_mwh", None)
            capex          = farm.get("capex_gbp_millions", None)

            depth_str        = f"{depth:.0f} m"           if depth is not None and pd.notna(depth)         else "N/A"
            capacity_str     = f"{capacity:.0f} MW"        if capacity is not None and pd.notna(capacity)   else "N/A"
            lcoe_turbine_str = f"£{lcoe_turbine:.1f}/MWh"  if lcoe_turbine is not None and pd.notna(lcoe_turbine) else "N/A"
            lcoe_flat_str    = f"£{lcoe_flat:.1f}/MWh"     if lcoe_flat is not None and pd.notna(lcoe_flat)       else "N/A"
            capex_str        = f"£{capex:.0f}M"            if capex is not None and pd.notna(capex)         else "N/A"

            popup_html = (
                f"<span style='color: purple; text-decoration: underline;'><b>{farm['name']}</b></span><br>"
                f"<i>{location}</i><br>"
                f"──────────────────<br>"
                f"<b>Turbine:</b> {turbine}<br>"
                f"<b>Foundation:</b> {fixed_flt}<br>"
                f"<b>Capacity:</b> {capacity_str}<br>"
                f"<b>Water depth:</b> {depth_str}<br>"
                f"──────────────────<br>"
                f"<b>Renewables.ninja CF:</b> {ninja_cf:.1%}<br>"
                f"<b>Calculated overall CF:</b> {calc_cf:.1%}<br>"
                f"<small>(After Array, Electrical &amp; Downtime Losses)</small><br>"
                f"──────────────────<br>"
                f"<b>Modelled CapEx (per turbine):</b> {capex_str}<br>"
                f"<b>Modelled LCOE (per turbine):</b> {lcoe_turbine_str}<br>"
                f"<small style='color:#555'>(15 MW basis, comparable to candidate sites)</small><br>"
                f"<b>LCOE — V2 flat CapEx:</b> {lcoe_flat_str}<br>"
                f"<small style='color:#555'>(£2.5M/MW assumption)</small>"
            )

            folium.CircleMarker(
                location=[farm["lat"], farm["lon"]],
                radius=7,
                color="#7b2d8b",
                fill=True,
                fill_color="#b15fcc",
                fill_opacity=0.85,
                weight=1.5,
                tooltip=farm["name"],
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(farms_fg)

        farms_fg.add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);
                font-family:sans-serif;font-size:12px;">
      <b>Suitability Score</b><br>
      <span style="color:#1a9641">■</span> &gt; 0.75 — Highly suitable<br>
      <span style="color:#a6d96a">■</span> 0.50–0.75 — Moderately suitable<br>
      <span style="color:#fdae61">■</span> 0.25–0.50 — Marginal<br>
      <span style="color:#d7191c">■</span> &lt; 0.25 — Poor suitability<br>
      <span style="color:#d7191c;opacity:0.4">■</span> Hard excluded<br>
      <span style="color:#b15fcc">●</span> Real UK wind farm
    </div>"""

    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = os.path.join(OUTPUT["maps_dir"], filename)
    m.save(out_path)
    print(f"[Vis] Suitability map saved -> {out_path}")
    return os.path.abspath(out_path)


#chart 1: capacity factor comparison

def make_capacity_factor_chart(sites_df: pd.DataFrame,
                                filename: str = "capacity_factors.html") -> str:
    """
    plotly bar chart comparing capacity factors across candidate sites.
    sorted lowest to highest so the best site appears at the top.
    """
    _ensure_output_dirs()

    if "mean_cf" not in sites_df.columns:
        print("[Vis] No mean_cf column — skipping capacity factor chart.")
        return ""

    df = sites_df.copy().sort_values("mean_cf", ascending=True)

    df["site_label"] = [
        f"Site {i}<br>({row['lat']:.1f}°N, {row['lon']:.1f}°E)"
        for i, (_, row) in enumerate(df.iterrows())
    ]

    fig = go.Figure(go.Bar(
        x=df["mean_cf"],
        y=df["site_label"],
        orientation="h",
        marker_color=df["mean_cf"],
        marker_colorscale="Viridis",
        text=[f"{cf:.1%}" for cf in df["mean_cf"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="Capacity Factor Comparison — Candidate Sites",
        xaxis_title="Mean Annual Capacity Factor",
        yaxis_title="Site",
        xaxis=dict(tickformat=".0%", range=[0, 0.70]),
        height=max(400, len(df) * 30 + 100),
        template="plotly_white",
        showlegend=False,
    )

    out_path = os.path.join(OUTPUT["charts_dir"], filename)
    fig.write_html(out_path)
    print(f"[Vis] Capacity factor chart saved -> {out_path}")
    return os.path.abspath(out_path)


#chart 2: lcoe comparison

def make_lcoe_chart(sites_df: pd.DataFrame,
                    filename: str = "lcoe_comparison.html") -> str:
    """
    plotly scatter plot of lcoe vs capacity factor, coloured by foundation type.

    bottom-right is best: high cf (lots of energy) + low lcoe (cheap to generate).
    bubble size represents aep - larger bubbles generate more energy per turbine per year.
    """
    _ensure_output_dirs()

    required = ["lcoe_gbp_per_mwh", "mean_cf", "foundation_type"]
    if not all(c in sites_df.columns for c in required):
        print("[Vis] Missing LCOE/CF columns — run economics analysis first.")
        return ""

    df = sites_df.dropna(subset=required).copy()

    fig = px.scatter(
        df,
        x="mean_cf",
        y="lcoe_gbp_per_mwh",
        color="foundation_type",
        size="aep_adjusted_mwh" if "aep_adjusted_mwh" in df.columns else None,
        hover_data=["lat", "lon", "capex_gbp_millions"],
        labels={
            "mean_cf":           "Mean Capacity Factor",
            "lcoe_gbp_per_mwh":  "LCOE (£/MWh)",
            "foundation_type":   "Foundation Type",
        },
        title="LCOE vs Capacity Factor by Site",
        color_discrete_map={"fixed-bottom": "#1f77b4", "floating": "#ff7f0e"},
        template="plotly_white",
    )

    fig.update_layout(xaxis_tickformat=".0%")

    out_path = os.path.join(OUTPUT["charts_dir"], filename)
    fig.write_html(out_path)
    print(f"[Vis] LCOE chart saved -> {out_path}")
    return os.path.abspath(out_path)


#chart 3: capacity factor time series

def make_cf_timeseries_chart(cf_series_dict: dict,
                              filename: str = "cf_timeseries.html") -> str:
    """
    plot monthly mean capacity factor for multiple sites on one chart.

    args:
        cf_series_dict: {site_label: pd.series of hourly cf values}
                        e.g. {"site 3 (55.2°n)": <series with datetimeindex>}
    """
    _ensure_output_dirs()

    fig = go.Figure()

    for label, series in cf_series_dict.items():
        monthly = series.resample("ME").mean()

        fig.add_trace(go.Scatter(
            x=monthly.index,
            y=monthly.values,
            mode="lines+markers",
            name=label,
        ))

    fig.update_layout(
        title="Monthly Mean Capacity Factor — Selected Sites",
        xaxis_title="Month",
        yaxis_title="Capacity Factor",
        yaxis_tickformat=".0%",
        template="plotly_white",
        legend_title="Site",
    )

    out_path = os.path.join(OUTPUT["charts_dir"], filename)
    fig.write_html(out_path)
    print(f"[Vis] CF time series chart saved -> {out_path}")
    return os.path.abspath(out_path)


#static figure exports (for dissertation)
#
#folium and plotly produce interactive html - these can't be embedded in a
#latex/word dissertation pdf. the functions below produce static pngs using
#geopandas and matplotlib.
#
#required: matplotlib, geopandas, contextily
#install:  pip install matplotlib geopandas contextily


def make_static_wind_map(wind_grid_df: pd.DataFrame,
                          eez_shapefile: str | None = None,
                          filename: str = "wind_resource_static.png",
                          dpi: int = 200) -> str:
    """
    export a static png of the wind resource grid for the dissertation.

    scatter plot coloured by mean wind speed (m/s) overlaid on the uk eez boundary.
    blue-red diverging palette so high-wind areas stand out clearly in print.

    args:
        wind_grid_df:  dataframe with 'lat', 'lon', 'mean_wind_ms' columns.
        eez_shapefile: optional path to eez .shp - draws the boundary outline.
        filename:      output png filename (saved to outputs/maps/).
        dpi:           image resolution. 200 dpi is fine for a4 at half-page width.

    returns:
        absolute path of the saved png file.
    """
    _ensure_output_dirs()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import geopandas as gpd
        from shapely.geometry import Point

        fig, ax = plt.subplots(figsize=(7, 9), facecolor="white")

        gdf = gpd.GeoDataFrame(
            wind_grid_df,
            geometry=gpd.points_from_xy(wind_grid_df["lon"], wind_grid_df["lat"]),
            crs="EPSG:4326"
        )

        #vmin/vmax pin the colour scale to realistic uk offshore wind speeds
        scatter = gdf.plot(
            ax=ax,
            column="mean_wind_ms",
            cmap="RdYlBu_r",      #blue = low wind, red = high wind
            markersize=8,
            vmin=6.0,
            vmax=13.0,
            legend=True,
            legend_kwds={
                "label":       "Mean Wind Speed (m/s)",
                "orientation": "vertical",
                "shrink":      0.65,
                "pad":         0.02,
            },
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )

        if eez_shapefile and os.path.isfile(eez_shapefile):
            eez = gpd.read_file(eez_shapefile).to_crs(epsg=4326)
            eez.boundary.plot(ax=ax, color="#0d47a1", linewidth=1.2,
                              label="UK EEZ", zorder=3)
            ax.legend(loc="lower left", fontsize=8)

        ax.set_title("Mean Offshore Wind Speed — UK EEZ\n"
                     "(Synthetic ERA5 placeholder, 0.5° resolution)",
                     fontsize=11, pad=10)
        ax.set_xlabel("Longitude (°)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_facecolor("#e8f4f8")   #pale blue ocean background

        plt.tight_layout()
        out_path = os.path.join(OUTPUT["maps_dir"], filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Vis] Static wind map saved -> {out_path}")
        return os.path.abspath(out_path)

    except Exception as e:
        print(f"[Vis] WARNING: Could not produce static wind map: {e}")
        return ""


def make_static_suitability_map(scored_df: pd.DataFrame,
                                 candidate_sites: pd.DataFrame = None,
                                 eez_shapefile: str | None = None,
                                 filename: str = "suitability_static.png",
                                 dpi: int = 200,
                                 v2_farms_df: pd.DataFrame = None) -> str:
    """
    export a static png of the suitability map for the dissertation.

    suitability scores on a red-yellow-green scale. hard-excluded points in grey.
    candidate sites marked with black stars.

    args:
        scored_df:       output of suitability.score_soft_constraints().
        candidate_sites: output of suitability.select_candidate_sites().
        eez_shapefile:   optional path to eez .shp for boundary overlay.
        filename:        output png filename (saved to outputs/maps/).
        dpi:             200 dpi recommended for a4 half-page.

    returns:
        absolute path of the saved png file.
    """
    _ensure_output_dirs()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        import geopandas as gpd

        fig, ax = plt.subplots(figsize=(7, 9), facecolor="white")
        ax.set_facecolor("#e8f4f8")

        #hard excluded points (grey0
        excluded = scored_df[scored_df.get("hard_excluded", pd.Series(False,
                             index=scored_df.index)) == True]
        if len(excluded):
            excl_gdf = gpd.GeoDataFrame(
                excluded,
                geometry=gpd.points_from_xy(excluded["lon"], excluded["lat"]),
                crs="EPSG:4326"
            )
            excl_gdf.plot(ax=ax, color="#bbbbbb", markersize=4, alpha=0.5, zorder=2)

        #scored points (rdylgn)
        scored = scored_df.dropna(subset=["suitability_score"])
        if len(scored):
            suit_gdf = gpd.GeoDataFrame(
                scored,
                geometry=gpd.points_from_xy(scored["lon"], scored["lat"]),
                crs="EPSG:4326"
            )
            suit_gdf.plot(
                ax=ax,
                column="suitability_score",
                cmap="RdYlGn",
                markersize=8,
                vmin=0.0,
                vmax=1.0,
                legend=True,
                legend_kwds={
                    "label":       "Suitability Score (0–1)",
                    "orientation": "vertical",
                    "shrink":      0.65,
                    "pad":         0.02,
                },
                zorder=3,
            )

        #eez boundary
        if eez_shapefile and os.path.isfile(eez_shapefile):
            eez = gpd.read_file(eez_shapefile).to_crs(epsg=4326)
            eez.boundary.plot(ax=ax, color="#0d47a1", linewidth=1.2, zorder=4)

        #candidate site stars
        if candidate_sites is not None and len(candidate_sites):
            cand_gdf = gpd.GeoDataFrame(
                candidate_sites,
                geometry=gpd.points_from_xy(candidate_sites["lon"],
                                             candidate_sites["lat"]),
                crs="EPSG:4326"
            )
            cand_gdf.plot(ax=ax, color="black", markersize=80,
                          marker="*", zorder=5, label="Candidate site")

            for idx, row in candidate_sites.iterrows():
                ax.annotate(
                    str(idx),
                    xy=(row["lon"], row["lat"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    zorder=6,
                )

        #real uk wind farms (v2 data)
        if v2_farms_df is not None and len(v2_farms_df):
            farms_gdf = gpd.GeoDataFrame(
                v2_farms_df,
                geometry=gpd.points_from_xy(v2_farms_df["lon"], v2_farms_df["lat"]),
                crs="EPSG:4326"
            )
            farms_gdf.plot(ax=ax, color="#b15fcc", markersize=40,
                           marker="^", zorder=6, label="Real UK wind farm")

        legend_patches = [
            mpatches.Patch(color="#bbbbbb", alpha=0.5, label="Hard excluded"),
            mpatches.Patch(color="black", label="Candidate site (★)"),
            mpatches.Patch(color="#b15fcc", label="Real UK wind farm (▲)"),
            mpatches.Patch(color="#0d47a1", label="UK EEZ boundary"),
        ]
        ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)

        ax.set_title("Offshore Wind Suitability — UK EEZ\n"
                     "(weighted composite: wind 40%, depth 25%, port 20%, seabed 15%)",
                     fontsize=11, pad=10)
        ax.set_xlabel("Longitude (°)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.tick_params(labelsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT["maps_dir"], filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Vis] Static suitability map saved -> {out_path}")
        return os.path.abspath(out_path)

    except Exception as e:
        print(f"[Vis] WARNING: Could not produce static suitability map: {e}")
        return ""


def make_static_lcoe_chart(sites_df: pd.DataFrame,
                            filename: str = "lcoe_static.png",
                            dpi: int = 200) -> str:
    """
    export a static png scatter plot of lcoe vs capacity factor for the dissertation.

    fixed-bottom and floating sites in different colours. best sites appear
    in the bottom-right quadrant (high cf, low lcoe).

    args:
        sites_df: dataframe with 'mean_cf', 'lcoe_gbp_per_mwh', 'foundation_type'.
        filename: output png filename (saved to outputs/charts/).
        dpi:      image resolution.

    returns:
        absolute path of the saved png file.
    """
    _ensure_output_dirs()

    required = ["lcoe_gbp_per_mwh", "mean_cf", "foundation_type"]
    if not all(c in sites_df.columns for c in required):
        print("[Vis] Missing LCOE/CF columns — skipping static LCOE chart.")
        return ""

    try:
        import matplotlib.pyplot as plt

        df = sites_df.dropna(subset=required).copy()

        colour_map = {"fixed-bottom": "#1f77b4", "floating": "#ff7f0e"}

        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")

        for ftype, group in df.groupby("foundation_type"):
            #scale bubble size by aep if available
            sizes = (group["aep_adjusted_mwh"] / group["aep_adjusted_mwh"].max() * 200
                     if "aep_adjusted_mwh" in group.columns
                     else 80)

            ax.scatter(
                group["mean_cf"],
                group["lcoe_gbp_per_mwh"],
                c=colour_map.get(ftype, "grey"),
                s=sizes,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=ftype.capitalize(),
                zorder=3,
            )

            for idx, row in group.iterrows():
                ax.annotate(
                    str(idx),
                    xy=(row["mean_cf"], row["lcoe_gbp_per_mwh"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color="black",
                )

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )

        ax.set_xlabel("Mean Annual Capacity Factor", fontsize=10)
        ax.set_ylabel("LCOE (£/MWh)", fontsize=10)
        ax.set_title("LCOE vs Capacity Factor — Candidate Sites\n"
                     "(bubble size proportional to annual energy production)",
                     fontsize=11)
        ax.legend(title="Foundation type", fontsize=9, title_fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9")

        plt.tight_layout()
        out_path = os.path.join(OUTPUT["charts_dir"], filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Vis] Static LCOE chart saved -> {out_path}")
        return os.path.abspath(out_path)

    except Exception as e:
        print(f"[Vis] WARNING: Could not produce static LCOE chart: {e}")
        return ""


def make_static_cf_chart(sites_df: pd.DataFrame,
                          filename: str = "cf_static.png",
                          dpi: int = 200) -> str:
    """
    export a static png horizontal bar chart of capacity factors for the dissertation.

    sites sorted so the highest cf appears at the top. bars coloured by viridis
    scale matching the interactive plotly version.

    args:
        sites_df: dataframe with 'mean_cf', 'lat', 'lon' columns.
        filename: output png filename (saved to outputs/charts/).
        dpi:      image resolution.

    returns:
        absolute path of the saved png file.
    """
    _ensure_output_dirs()

    if "mean_cf" not in sites_df.columns:
        print("[Vis] No mean_cf column — skipping static CF chart.")
        return ""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        df = sites_df.copy().sort_values("mean_cf", ascending=True).reset_index()

        labels = [
            f"Site {row['site_id'] if 'site_id' in row else i}  "
            f"({row['lat']:.1f}°N, {row['lon']:.1f}°E)"
            for i, (_, row) in enumerate(df.iterrows())
        ]

        norm = mcolors.Normalize(vmin=df["mean_cf"].min(), vmax=df["mean_cf"].max())
        colours = cm.viridis(norm(df["mean_cf"].values))

        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.45 + 1)),
                               facecolor="white")

        bars = ax.barh(labels, df["mean_cf"], color=colours, edgecolor="white",
                       linewidth=0.5)

        #percentage labels to the right of each bar
        for bar, cf in zip(bars, df["mean_cf"]):
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{cf:.1%}",
                va="center", ha="left", fontsize=8,
            )

        ax.set_xlabel("Mean Annual Capacity Factor", fontsize=10)
        ax.set_title("Annual Capacity Factor — Candidate Sites", fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlim(0, df["mean_cf"].max() + 0.08)
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor("#f9f9f9")
        ax.tick_params(labelsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT["charts_dir"], filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Vis] Static CF chart saved -> {out_path}")
        return os.path.abspath(out_path)

    except Exception as e:
        print(f"[Vis] WARNING: Could not produce static CF chart: {e}")
        return ""