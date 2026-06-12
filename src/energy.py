#src/energy.py - annual energy production (aep) estimation
#aep = capacity (kw) × 8760 × mean capacity factor
#relies on hourly cf time series from the renewables.ninja api (via data_loader.py).

import numpy as np
import pandas as pd


def estimate_cf_from_wind_speed(mean_wind_ms: float,
                                 rated_power_mw: float = 15) -> float:
    """
    estimate capacity factor from mean annual wind speed using an empirical
    relationship based on rayleigh wind distribution and typical offshore
    turbine power curves. used when hourly api data is unavailable.

    args:
        mean_wind_ms:   mean annual wind speed in m/s
        rated_power_mw: turbine rated power in mw (default 15 mw)

    returns:
        estimated capacity factor (0-1)
    """
    if mean_wind_ms < 5.0:
        #below cut-in: very low output
        return max(0.0, (mean_wind_ms - 3.5) * 0.05)
    elif mean_wind_ms < 15.0:
        #linear region: ~5% cf gain per m/s
        return 0.18 + (mean_wind_ms - 5.0) * 0.046
    else:
        #above rated speed: power is capped, slow growth
        cf_at_15 = 0.18 + (15.0 - 5.0) * 0.046
        return cf_at_15 + (mean_wind_ms - 15.0) * 0.008


def apply_losses_to_cf(mean_cf: float,
                       array_loss_pct: float = 11.27,
                       electrical_loss_pct: float = 8.90,
                       downtime_pct: float = 5.97) -> float:
    """
    apply operational losses to a raw mean capacity factor.

    formula (from v2 excel data):
        calculated cf = mean cf × (1 - array loss) × (1 - electrical loss) × (1 - downtime)

    args:
        mean_cf:              raw mean annual capacity factor (0-1)
        array_loss_pct:       wake + inter-array cable losses (%)
        electrical_loss_pct:  substation + export cable losses (%)
        downtime_pct:         maintenance, outages, curtailment (%)

    returns:
        capacity factor after all losses (0-1)
    """
    array_loss_dec = array_loss_pct / 100.0
    elec_loss_dec = electrical_loss_pct / 100.0
    downtime_dec = downtime_pct / 100.0

    calculated_cf = mean_cf * (1 - array_loss_dec) * (1 - elec_loss_dec) * (1 - downtime_dec)
    return round(calculated_cf, 4)


def compute_aep(capacity_factor_series: pd.Series,
                rated_power_mw: float,
                availability: float = 0.94) -> dict:
    """
    compute annual energy production from a capacity factor time series.

    aep formula:
        aep_raw      = rated_power_mw × hours × mean_cf   (mwh)
        aep_adjusted = aep_raw × availability              (mwh)

    availability (default 0.94) accounts for maintenance, downtime, and curtailment
    - sourced from crabtree et al. (2015).

    args:
        capacity_factor_series: hourly cf values (0-1), typically 8760 values.
        rated_power_mw:         turbine rated power in mw.
        availability:           turbine availability factor (default 0.94).

    returns:
        dict: mean_cf, aep_mwh, aep_adjusted_mwh, capacity_factor_p50, hours_above_50pct
    """
    cf = capacity_factor_series.dropna()  #drop nans - handles partial years or api gaps
    mean_cf = float(cf.mean())
    hours = len(cf)  #use actual hours rather than assuming 8760

    aep_raw = rated_power_mw * hours * mean_cf
    aep_adj = aep_raw * availability

    return {
        "mean_cf":             round(mean_cf, 4),
        "aep_mwh":             round(aep_raw, 1),
        "aep_adjusted_mwh":    round(aep_adj, 1),
        "capacity_factor_p50": round(float(cf.quantile(0.50)), 4),
        "hours_above_50pct":   int((cf > 0.5).sum()),
    }


def compute_aep_for_all_sites(candidate_sites: pd.DataFrame,
                               ninja_token: str = "",
                               year: int = 2019,
                               v2_farms_df=None) -> pd.DataFrame:
    """
    fetch capacity factor time series and compute aep for every candidate site.

    iterates over rows in candidate_sites (must have 'lat', 'lon' columns).
    caches api responses locally to avoid rate-limiting (see data_loader.py).

    year 2019 is used as it's the most recent complete era5 year and avoids
    covid-related anomalies in 2020.

    args:
        candidate_sites: dataframe with 'lat' and 'lon' columns.
        ninja_token:     renewables.ninja api token (leave blank for synthetic data).
        year:            calendar year to fetch hourly cf data for.

    returns:
        candidate_sites with mean_cf, aep_mwh, aep_adjusted_mwh,
        capacity_factor_p50, hours_above_50pct columns appended.
    """
    #deferred imports to avoid circular dependency with config.py
    from src.data_loader import fetch_ninja_capacity_factors
    from config import TURBINE

    results = []

    for idx, site in candidate_sites.iterrows():
        cf_series = fetch_ninja_capacity_factors(
            lat=site["lat"],
            lon=site["lon"],
            token=ninja_token,
            year=year,
            v2_farms_df=v2_farms_df
        )

        aep = compute_aep(
            cf_series,
            rated_power_mw=TURBINE["rated_power_mw"],
            availability=TURBINE["availability"]
        )

        results.append(aep)

        print(f"  Site {idx}: mean CF={aep['mean_cf']:.3f}, "
              f"AEP={aep['aep_adjusted_mwh']:,.0f} MWh/turbine/yr")

    aep_df = pd.DataFrame(results, index=candidate_sites.index)
    return pd.concat([candidate_sites, aep_df], axis=1)


def flag_dunkelflaute(capacity_factor_series: pd.Series,
                       threshold_cf: float = 0.10,
                       min_duration_hours: int = 48) -> pd.DataFrame:
    """
    identify dunkelflaute events: prolonged periods of very low wind output.

    "dunkelflaute" describes multi-day spells where wind speeds are too low for
    meaningful generation - typically winter anticyclones. concept from
    potisomporn & vogel (2022).

    single linear scan: tracks whether we're inside a low-cf event and records
    it when it ends, provided it meets the minimum duration threshold.

    args:
        capacity_factor_series: hourly cf time series (pd.series with datetimeindex).
        threshold_cf:           cf below this counts as "low wind" (default 0.10).
        min_duration_hours:     minimum consecutive low-wind hours to qualify (default 48 h).

    returns:
        dataframe of events with columns: start, end, duration_hours, mean_cf
    """
    cf = capacity_factor_series.copy()
    low = cf < threshold_cf

    events = []
    in_event = False
    start = None

    for ts, val in low.items():
        if val and not in_event:
            in_event = True
            start = ts
        elif not val and in_event:
            duration = int((ts - start).total_seconds() / 3600)
            if duration >= min_duration_hours:
                events.append({
                    "start":          start,
                    "end":            ts,
                    "duration_hours": duration,
                    "mean_cf":        float(cf[start:ts].mean()),
                })
            in_event = False

    print(f"[Energy] Found {len(events)} Dunkelflaute events "
          f"(>{min_duration_hours} h below CF {threshold_cf}).")

    return pd.DataFrame(events)