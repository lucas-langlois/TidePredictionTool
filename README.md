# CSIRO Tide Prediction Tool

Desktop application for tidal prediction using CSIRO tidal constants and UTide harmonic reconstruction.

## Important Runtime Requirement

This tool will not run predictions unless the CSIRO model file is present:

- Required file: `CSIRO_tidal_const_v12.nc`
- Download: https://data.csiro.au/collection/csiro:45584
- Location for Python run: same folder as `TidePredictionApp.py`
- Location for EXE run: same folder as `TidePredictionTool.exe`

Note: The `.nc` file is intentionally not stored in this repository.

## What This Tool Does

The app predicts tide height time series from the CSIRO model file `CSIRO_tidal_const_v12.nc`.

It supports three workflows:

1. Single Location: run one coordinate over a date range and plot/export results.
2. Survey Location: use a polygon area (imported or drawn), predict all mesh faces inside it, and export the area-average tide series.
3. Batch Processing: process many rows from CSV in either time-period mode or single-time mode.

## Current UI Tabs

1. Single Location
2. Survey Location
3. Batch Processing

## Core Features

- PySide6 desktop UI
- Interactive map coordinate picker (single location)
- Interactive native map polygon drawing (survey)
- KML and SHP polygon import for survey mode
- CRS handling for SHP:
  - If `.prj` exists and CRS is not EPSG:4326, geometry is transformed to EPSG:4326
  - If `.prj` is missing, data must already look like lon/lat or the tool raises an error
- Multiple Australian timezone support
- CSV export with `Date_Time` as `YYYY-MM-DD HH:MM:SS`
- Batch mode auto-clears form when switching mode

## Requirements

### Required Files

- `TidePredictionApp.py`
- `CSIRO_tidal_const_v12.nc` (required at runtime, not committed to this repo)

### Download CSIRO Model File

Download `CSIRO_tidal_const_v12.nc` from:

- https://data.csiro.au/collection/csiro:45584

After download, place the file in the same folder as `TidePredictionApp.py` (or beside `TidePredictionTool.exe` for distribution runs).

This repository intentionally excludes the `.nc` file from Git history.

For release distribution:

- You may also exclude the `.nc` file from the release zip.
- End users must download it from the same CSIRO link and place it beside the exe.

### Python Packages

- numpy
- pandas
- scipy
- matplotlib
- xarray
- utide
- pytz
- PySide6
- pyshp (module name: `shapefile`) for SHP support
- pyproj for SHP CRS transformation to EPSG:4326

## Suggested Environment Setup

```bash
conda create -n csiro_tides python=3.10 -y
conda activate csiro_tides
conda install numpy pandas scipy matplotlib xarray -y
pip install utide pytz PySide6 pyshp pyproj
```

## Run the App

```bash
python TidePredictionApp.py
```

Important: `CSIRO_tidal_const_v12.nc` must be present in the same folder before running.

The app creates/uses:

- `input/` for input CSV files
- `prediction_outputs/` for outputs

## Usage

### 1) Single Location

Use when you want one location and one time range.

Steps:

1. Enter longitude and latitude, or click `Pick on Map`.
2. Enter start/end dates (`YYYY-MM-DD`).
3. Choose frequency and timezone.
4. Run analysis.
5. Save CSV / plot PNG / summary JSON if needed.

Output:

- `prediction_outputs/<site>.csv`

### 2) Survey Location

Use when you want area-averaged tides across a polygon.

Inputs:

- Survey name
- Start/end dates
- Frequency
- Timezone
- Polygon source:
  - Import KML/SHP, or
  - Draw polygon on map

How results are computed:

1. Identify model mesh faces whose centroids fall within the survey polygon.
2. Reconstruct tide series for each selected face.
3. Average tide values across faces at each time step.

Output:

- `prediction_outputs/<survey_name>_survey_average.csv`
- Includes `n_faces_averaged`

### 3) Batch Processing

Two modes:

1. Time Period Predictions
2. Single Time Point Predictions

Important behavior:

- Switching between these two modes clears loaded batch form state to prevent mixing incompatible inputs.

#### Time Period CSV

Required columns:

- `Site,Longitude,Latitude,start,stop,interval`

Output:

- One CSV per site in `prediction_outputs/`

#### Single Time CSV

Required columns:

- `Site,Latitude,Longitude,Date_Time`

Output:

- `prediction_outputs/tide_predictions_single_time.csv`

Supported date-time formats include:

- `DD/MM/YYYY HH:MM:SS`
- `DD/MM/YYYY HH:MM`
- `DD/MM/YYYY`
- `YYYY-MM-DD HH:MM:SS`
- `YYYY-MM-DD HH:MM`
- `YYYY-MM-DD`

## Exe Build

Use `build_exe.py`.

Bundled hidden imports include:

- `shapefile` (pyshp)
- `pyproj`

This ensures SHP read + CRS transformation works in the packaged executable.

Runtime requirement for EXE:

- Keep `CSIRO_tidal_const_v12.nc` in the same directory as `TidePredictionTool.exe`.
- The exe will not run predictions without that file.

## GitHub Release Usage

If you download `TidePredictionTool.exe` from Releases:

1. Download `CSIRO_tidal_const_v12.nc` from https://data.csiro.au/collection/csiro:45584
2. Put `CSIRO_tidal_const_v12.nc` in the same folder as `TidePredictionTool.exe`
3. Run `TidePredictionTool.exe`

Build outputs are organized under:

- `artifacts/pyinstaller/`
  - `build/`, `dist/`, `spec/`
- `artifacts/releases/`
  - release-ready distribution folders/zips

## Troubleshooting

### NetCDF file not found

Ensure `CSIRO_tidal_const_v12.nc` is beside the app/exe.
If missing, download it from: https://data.csiro.au/collection/csiro:45584

### SHP import fails with CRS message

- Include the `.prj` file next to the `.shp`.
- Or provide geometry already in EPSG:4326 lon/lat.

### No faces found in survey polygon

- Verify polygon overlaps the model domain.
- Verify polygon coordinates are in EPSG:4326.

### Map interaction feels odd

Recent versions include improved pan/zoom smoothing and request deduplication.

## Project Structure

```text
Tide_Prediction_Tool/
  TidePredictionApp.py
  build_exe.py
  artifacts/
    pyinstaller/
    releases/
  CSIRO_tidal_const_v12.nc  # download locally; not tracked in Git
  README.md
  QUICK_START_INSTRUCTIONS.txt
  input/
  prediction_outputs/
```
