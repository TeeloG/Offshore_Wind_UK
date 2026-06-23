<p align="center">
  <img src="paper/ukowepipe_logo.svg" width="270" alt="logo">
</p>

<h1 align="center">United Kingdom Offshore Wind Pipeline</h1>

<p align="center">
  An open, reproducible Python pipeline for siting, energy-yield, and cost
  assessment of offshore wind across the UK Exclusive Economic Zone.
</p>

---

## What it does

From open data, in a single run:

1. **Map** UK waters on a grid.
2. **Score & select** the best sites — wind, water depth, port distance, seabed.
3. **Estimate energy** — capacity factor and annual yield from ERA5 reanalysis.
4. **Estimate cost** — levelised cost of energy (LCOE) per site.
5. **Flag low-wind risk** — multi-day "Dunkelflaute" events.

Produces interactive maps and charts plus publication-ready figures.

## Data (all open)

ERA5 (wind) · GEBCO (bathymetry) · JNCC (protected areas & EEZ) · BGS (seabed).

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Status

🚧 Work in progress, name is provisional. Results are being validated against
44 operational UK offshore wind farms.
