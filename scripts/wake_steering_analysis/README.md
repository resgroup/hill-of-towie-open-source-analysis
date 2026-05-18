# Hill of Towie open dataset (v2)

Operational and remote-sensing data from Hill of Towie, a 21-turbine
Siemens SWT-2.3-VS-82 wind farm in Scotland. Version 2 extends the v1
release (2016 → 2024-08) to 2026-04 and adds three new data tiers:
high-resolution turbine fastlog, two LiDARs, and telemetry from the
Dynamic Yaw wind farm controller tested Jan–Apr 2026.

## Contents

| File | Description | Coverage |
| --- | --- | --- |
| `2016.zip` … `2025.zip` | 10-minute SCADA archives, one zip per year | full years 2016 – 2025 |
| `2026.zip` | 10-minute SCADA, partial year | 2026-01 – 2026-04 |
| `Hill_of_Towie_turbine_metadata.csv` | Name, Station ID, lat/lon, model, hub height, rotor diameter, COD | — |
| `turbine_fastlog.zip` | High-resolution turbine SCADA and Dynamic Yaw controller telemetry | 2026-01-01 – 2026-04-30 |
| `lidar_data.zip` | ZX300 ground LiDAR + ZXTM nacelle LiDAR on T07 | 2025-08-21 – 2026-04-30 (see §LiDAR) |
| `ZXMS-RES029-REP01-01.pdf` | ZX300 deployment report | Aug 2025 |
| `ZXMS-RES035-REP01-01.pdf` | ZXTM deployment report | Dec 2025 |
| `README.md` | This file | — |

### Supporting metadata and data dictionaries

These small reference files are carried over unchanged from the v1 release
([zenodo.org/records/14870023](https://zenodo.org/records/14870023)) and
cover the 10-minute SCADA data. The fastlog and LiDAR tiers added in v2 are
documented in this README and in the companion code.

| File | Description |
| --- | --- |
| `Hill_of_Towie_tables_description.csv` | One row per SCADA table contained in the year zips |
| `Hill_of_Towie_turbine_fields_description.csv` | Definitions, source table, and units for selected `wtc_*` 10-minute fields (a subset, not exhaustive) |
| `Hill_of_Towie_grid_fields_description.csv` | Definitions, source table, and units for the 10-minute fields in the `tblGrid*` tables (a subset, not exhaustive) |
| `Hill_of_Towie_alarms_description.csv` | Lookup mapping selected `Alarmcode` values in `tblAlarmLog` to descriptions and a stopping/non-stopping flag |
| `Hill_of_Towie_AeroUp_install_dates.csv` | Per-turbine first and last AeroUp retrofit work dates |
| `Hill_of_Towie_ShutdownDuration.zip` | Computed shutdown durations covering the v1 data window based on SCADA data and RES operator logs |

## Turbines

Hill of Towie consists of 21 Siemens SWT-2.3-VS-82 turbines (2.3 MW, 59 m hub,
82 m rotor). Turbines are named `T01` through
`T21`. The SCADA join key is `StationId` (a.k.a. `StationNr`), which equals
`wtg_number + 2304509` (so T01 = 2304510, T21 = 2304530). Coordinates are in `Hill_of_Towie_turbine_metadata.csv`.

## 10-minute SCADA (`{YEAR}.zip`)

Each zip contains 13 monthly CSV tables named `{table}_{YYYY}_{MM}.csv`.
Primary tables:

- `tblSCTurbine` — ~80 fields per turbine per 10-minute period, named
  `wtc_*_{min,max,mean,stddev,endvalue,timeon}` (e.g. `wtc_ActPower_mean`,
  `wtc_AcWindSp_mean`, `wtc_NacelPos_mean`, `wtc_PitcPosA_mean`,
  `wtc_AmbieTmp_mean`, `wtc_ScReToOp_timeon`, `wtc_ScYawOpe_counts`).
- `tblAlarmLog` — alarm events with `TimeOn`, `TimeOff`, `StationNr`, `Alarmcode`.
- Supporting tables: `tblDailySummary`, `tblGrid`, `tblGridScientific`,
  `tblSCTurCount`, `tblSCTurDigiIn`, `tblSCTurDigiOut`, `tblSCTurFlag`, `tblSCTurGrid`,
  `tblSCTurIntern`, `tblSCTurPress`, `tblSCTurTemp`.

`TimeStamp` columns are **end-of-period UTC** — subtract 10 minutes for the
start-of-period timestamp. Rows are joined to turbine metadata on
`StationId` / `StationNr`.

## Turbine fastlog (`turbine_fastlog.zip`)

After extracting:

```
Filestore/FL/HOT/{station_nr}/{YYYY-MM-DD}/FL{station_nr}_{TAG}_{YYYY_MM_DD}.prq
```

The `.prq` files are Apache Parquet — the extension is just a legacy
convention; readers should treat them exactly like `.parquet` files.

One parquet file per signal, per turbine, per day. Coverage is
2026-01-01 → 2026-04-30 for all 21 turbines. Each parquet has two columns:
`timestamp` (microsecond UTC) and the named signal value. Native sampling is
approximately 1-25 Hz (signal dependent), log on change.

The Hill of Towie turbine controllers record thousands of tags internally;
this data pack contains only a curated subset chosen for research value and
archive size. Two tiers are provided:

**Dynamic Yaw tags — present on all 21 turbines (5 tags per turbine per
day):**

- `computed_core_post_processed_consensus_wind_direction_true_degrees` —
  the Dynamic Yaw controller's per turbine wind-direction estimate, referenced to true north.
- `computed_core_post_processed_core_wake_steering_offset_degrees` — the Dynamic Yaw
  commanded wake steering yaw offset (signed, +ve means CCW viewed from above).
- `computed_driver_post_processed_toggle_state` — 0 when the Dynamic Yaw
  controller is not active (off), 1 when active (on). Defines the on/off
  windows used for uplift analysis.
- `computed_driver_post_processed_yaw_target_degrees` — the Dynamic Yaw commanded yaw
  target in true-north reference. Includes contributions from collective yaw control and wake steering.
- `computed_driver_pre_processed_yaw_direction_true_degrees` — measured
  yaw direction with the Dynamic Yaw controller's northing correction already applied.

**Extended Siemens tags — only on T01, T02, T03, T04, T05, T07, T11, T13, T14
(the western turbines near wake-steering activity). 15 additional tags per
turbine per day:**

- Power and setpoint tags: `Wtc_TDI_ActPower_Value`, `Wtc_TDI_ActLimit_Power`,
  `Wtc_TDI_PowerRef_PowerRef`, `Wtc_TDI_PowerRed_PowerRed`,
  `Wtc_TDI_ReactPwr_Value`.
- Wind sensing: `Wtc_TDI_AcWindSp_AcWindSp`, `Wtc_TDI_AcWindDr_Value`,
  `Wtc_TDI_AcWindDr_Source`.
- Pitch: `Wtc_TDI_PitcPosA_Value`, `Wtc_TDI_PitcPosB_Value`,
  `Wtc_TDI_PitcPosC_Value`.
- Yaw: `Wtc_TDI_YawPos_Value`, `Wtc_TDI_YawExec_YawExec`.
- Drivetrain / state: `Wtc_TDI_MainSRpm_Value`, `Wtc_TDI_GenState_GenState`.

## LiDAR (`lidar_data.zip`)

After extracting:

```
lidar_data/timeseries/{unit_id}/Wind10_{prefix}@Y{YYYY}_M{MM}_D{DD}.parquet
lidar_data/timeseries/{unit_id}/Wind_{prefix}@Y{YYYY}_M{MM}_D{DD}.parquet
```

All LiDAR data files are Apache Parquet (same format as the `.prq` files in
the turbine fastlog — only the extension differs). There are two file types
per instrument per day: `Wind10_*` is 10-minute aggregated, `Wind_*` is
high-resolution.

**ZX300 (unit 2428) — ground-based vertical profile LiDAR.** Located ~280 m
north of T11. Columns include `Horizontal Wind Speed (m/s) at {h}m`,
`Wind Direction (deg) at {h}m`, and `Vertical Wind Speed (m/s) at {h}m` for
a series of fixed heights, plus met station and status columns
(`Met Compass Bearing (deg)`, `Met Air Temp. (C)`, `Met Pressure (mbar)`,
`Met Humidity (%)`, `Battery (V)`, `Info. Flags`, `Status Flags`). Hub
height = 58 m. Bad values are written as sentinel values from `9989` to
`9999`.

**ZXTM (unit 5060) — nacelle-mounted cone-scanning LiDAR on T07.** Scans
forward of the rotor with a 15° half-angle. Each row carries a `Range (m)`
value in {10, 48, 64, 85, 126, 208} — distance in front of T07 along the
beam axis. Line-of-sight wind speeds are in columns named
`{Left|Right} LOS Speed (m/s) at Rotor Segment Height {h}m` with heights
{26.2, 42.6, 59.0, 75.4, 91.8} m above the turbine's ground level.
`Met Compass Bearing (deg)` is the instrument's raw compass reading and
needs a northing correction (see §Conventions). Bad values are written as
sentinel values from `9987` to `9999`.

**Pair-derived (PD) outputs** — `PD Wind Yaw Misalignment (deg) at Hub Height`
and `PD Horizontal Wind Speed (m/s) at Hub Height` are computed by resolving
the left and right line-of-sight values at hub height under an assumption of
uniform flow. In uniform conditions this is the most accurate measure of the
incoming wind at hub height.

**Fit-derived (FD) outputs** — `FD Horizontal Wind Speed (m/s) at Hub Height`,
`FD Wind Yaw Misalignment (deg)`, `FD Mean Fit Residual Normalised By Wind Speed`,
and `FD Vertical Wind Shear Exponent` are produced by fitting a few seconds of
line-of-sight data to a model assuming uniform flow with power-law shear and
no veer. The fit residuals quantify inflow complexity and can be used to
filter both the fit-derived and pair-derived outputs.

**Coverage:**

| Instrument | `Wind10_*` (10-min) | `Wind_*` (high-resolution) |
| --- | --- | --- |
| ZX300 (2428) | 2025-08-21 – 2026-04-30 | 2026-01-01 – 2026-04-30 |
| ZXTM (5060) | 2025-12-12 – 2026-04-30 | 2026-01-01 – 2026-04-30 |

High-resolution files are only provided for the Dynamic Yaw campaign window
(Jan – Apr 2026) to keep the archive size manageable. Configuration and
installation detail is in `ZXMS-RES029-REP01-01.pdf` (ZX300) and
`ZXMS-RES035-REP01-01.pdf` (ZXTM).

## Conventions

- All timestamps are UTC.
- The 10-minute SCADA `TimeStamp` is end-of-period — subtract 10 minutes
  for the start.
- Angles are in degrees, 0° = true north, increasing clockwise.
- Wind direction is the meteorological convention: the direction the wind
  blows **from**.
- `wtc_NacelPos_mean`, `Wtc_TDI_YawPos_Value`, and
  `Met Compass Bearing (deg)` are **raw sensor readings** and require a
  per-instrument northing calibration to be aligned with true north. The
  Dynamic Yaw tags whose names end in `_true_degrees` are already northed.
  Northing values used during the Dynamic Yaw campaign are published in
  the companion code.

## Companion code

Open-source Python code for loading every data tier in this archive is
available at:

**https://github.com/resgroup/hill-of-towie-open-source-analysis**

## License and citation

This dataset is released by RES on behalf of TRIG under a Creative Commons
Attribution 4.0 (CC-BY-4.0) license and is provided as-is. If you use it,
please cite the Zenodo record https://doi.org/10.5281/zenodo.20204946.

Contact: Alex Clerc — `Alex.Clerc@res-group.com`
For ZX Lidars enquires use `support@zxlidars.com`
