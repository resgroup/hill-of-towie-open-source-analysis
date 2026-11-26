# WeDoWind wake steering challenge: participant instructions

Hill of Towie (HOT) is a 21-turbine wind farm in Scotland where some turbines steer their wakes away from a
downstream neighbour by yawing off the wind. The challenge is to predict what a downstream turbine does (and where there is LiDAR, what the wind speed is) from its
upstream neighbour's data alone.

## The task

Each scoring case is a pair: an **upstream turbine** and a **downstream turbine** in its wake. For each of ten
scoring periods you are given 1 Hz data from the pair's upstream turbine and nothing from any other turbine. For
every minute of every period, predict:

- the minute-mean power of the pair's **downstream turbine**, in kW;
- for the T03 → T07 pair only, the minute-mean line-of-sight (LOS) wind speed of two beams of the nacelle LiDAR
  on T07, which looks upwind towards T03's wake.

Any method is welcome: an engineering wake model, CFD, machine learning, a lookup table, or a mix. You may use
anything in the public datapack to build and tune it.

| Pair | Upstream turbine is at | Spacing | Scoring periods (UTC, end exclusive) | Minutes |
|---|---|---|---|---|
| T03 → T07 | 209° from T07 | 389 m, 4.7 D | 2026-09-01 02:00 – 04:30 | 150 |
| | | | 2026-09-04 03:00 – 07:00 | 240 |
| | | | 2026-10-04 00:00 – 04:00 | 240 |
| | | | 2026-11-14 13:00 – 19:30 | 390 |
| T11 → T14 | 238° from T14 | 404 m, 4.9 D | 2026-07-30 23:30 – 2026-07-31 07:00 | 450 |
| | | | 2026-07-31 22:00 – 2026-08-01 10:00 | 720 |
| | | | 2026-10-15 04:00 – 09:30 | 330 |
| T07 → T08 | 276° from T08 | 501 m, 6.1 D | 2026-10-10 20:00 – 2026-10-11 00:00 | 240 |
| T07 → T04 | 327° from T04 | 419 m, 5.1 D | 2026-07-18 21:00 – 2026-07-19 00:00 | 180 |
| | | | 2026-08-20 15:30 – 2026-08-21 00:00 | 510 |

"209° from T07" means T03 is on a bearing of 209° from T07, so T07 sits in the centre of T03's
wake when the wind blows from 209°. These bearings are from the turbine positions only; the direction signals
can sit a few degrees away from them (see [Things to know about the data](#things-to-know-about-the-data)). D is
the rotor diameter. That is 3,450 minutes in all, 1,020 of them in the
T03 → T07 pair.

## The data

### Training data: the public datapack

[Hill of Towie open dataset v2.1.0](https://zenodo.org/records/22662930) on Zenodo. Its `README.md` and
`Hill_of_Towie_data_dictionary.md` describe every file and column; read the dictionary's sections on reference
frames and on the fastlog. The parts that matter here:

- `turbine_fastlog.zip`: about 1 Hz turbine data for 2026-01-01 to 2026-04-30. All 21 turbines have the five
  Dynamic Yaw (DY) controller tags; T01, T02, T03, T04, T05, T07, T11, T13 and T14 also have 15 turbine tags
  (power, wind speed, yaw position and so on). **T08 has no 1 Hz power in the datapack**; its power is in the
  10-minute SCADA.
- `lidar_data.zip`: the ZXTM nacelle LiDAR on T07 (unit 5060), 2025-12-12 to 2026-04-30, and a ZX300 ground LiDAR.
- `2016.zip` … `2026.zip`: 10-minute SCADA for every turbine, to 2026-04.
- `Hill_of_Towie_turbine_metadata.csv`: turbine positions. Every turbine is a Siemens SWT-2.3-VS-82: 2,300 kW
  rated, 82 m rotor diameter, 59 m hub height. No manufacturer power or thrust curve is published; the worked
  example uses a generic one (`floris_inputs/swt_2_3_82_generic.yaml`).

The loaders in this repository download what they need from Zenodo on first use (several GB for the fastlog).

In the datapack period the DY controller was being validated with a toggle: the tag
`computed_driver_post_processed_toggle_state` is 1 when the controller is in charge (steering allowed) and 0 when
it is not, switching back and forth through the day. Toggle-off and toggle-on minutes in similar wind are the
cleanest view of what steering does. In the datapack period T03 and T11 steer. **T07 never steers in the datapack
period** (its commanded steering offset is always zero), so for the two T07 pairs the public data shows the
unsteered wake only.

### Scoring-day inputs

> **Placeholder:** the download location of the scoring-day inputs will be added here before the challenge opens.

One folder, `turbine_fastlog/Filestore/FL/HOT/<station number>/<date>/`, in the same layout and file format as the
datapack fastlog. It holds the upstream turbine's files for its own periods: T03 (station 2304512) for the
T03 → T07 days, T11 (2304520) for the T11 → T14 days and T07 (2304516) for the T07 → T08 and T07 → T04 days.
The station number is the turbine number plus 2304509.

- Each period's days have all 15 turbine tags and two of the DY tags: the consensus wind direction and the
  measured yaw direction.
- The day before each period is included too. Most tags are logged only when they change, so the previous day is
  needed to know a tag's value at midnight.
- The day after each period has three status tags only (`GenState_GenState`, `ActLimit_Power`,
  `PowerRed_PowerRed`), for the same reason at the other end of the day.
- **The other three DY tags are not included, and the control in the scoring periods is not stated.** The
  toggle, the wake steering offset and the yaw target are left out: the wake does not feel a command, only where
  the nacelle points. Wake steering may or may not be on, and if it is on it may be calibrated differently from the open data.

You get no data from the downstream turbine, from any other turbine or from either LiDAR for the scoring days.
The 10-minute SCADA in the datapack also ends before them.

To read either the datapack fastlog or the scoring-day inputs at 1 Hz:

```python
import pandas as pd
from pathlib import Path
from hot_open.fastlog_helpers import load_hot_fl_data

fl_df = load_hot_fl_data(
    data_dir=Path("<scoring inputs>/turbine_fastlog/Filestore"),
    wtg_numbers=[3],
    start_dt=pd.Timestamp("2026-09-01 00:00", tz="UTC"),
    end_dt_excl=pd.Timestamp("2026-09-01 04:40", tz="UTC"),
    extra_tags=[
        "computed_core_post_processed_consensus_wind_direction_true_degrees",
        "computed_driver_pre_processed_yaw_direction_true_degrees",
    ],
)
fl_df["T03"]["ActPower_Value"]
```

The result has a `(turbine, tag)` column MultiIndex on a 1 s UTC index; the turbine tags lose their `Wtc_TDI_`
prefix. Log-on-change tags are forward-filled, but not across a pause in logging. Start a load at midnight if
you need the status tags: they always have a value there. The loader caches by turbine and day and not by folder, so
use a separate cache directory (`HOT_OPEN_CACHE_DIR`) if you ever keep two different copies of the same
turbine-day. The first load of a turbine-day takes the best part of a minute; after that it comes from the cache.
For months of training data it is quicker to read the tag files yourself: each is a parquet file indexed by
`timestamp` with one value column.

### Tags that matter

| Tag | Meaning | In scoring inputs |
|---|---|---|
| `Wtc_TDI_ActPower_Value` | Active power, kW. | yes |
| `Wtc_TDI_AcWindSp_AcWindSp` | Nacelle anemometer wind speed, m/s. Behind the rotor, so not the free-stream speed. | yes |
| `Wtc_TDI_AcWindDr_Value` | Wind direction from the turbine's own wind sensor, in the turbine's own north frame, not true north. | yes |
| `Wtc_TDI_YawPos_Value` | Yaw position in the turbine's own north frame, not true north. | yes |
| `Wtc_TDI_MainSRpm_Value` | Main shaft speed, rpm. | yes |
| `Wtc_TDI_PitcPosA_Value`, `B`, `C` | Blade pitch angles, degrees. | yes |
| `computed_core_post_processed_consensus_wind_direction_true_degrees` | The DY controller's estimate of the wind direction at this turbine, drawing on several turbines. Degrees clockwise from true north, direction the wind comes from. | yes |
| `computed_driver_pre_processed_yaw_direction_true_degrees` | Measured yaw direction (where the rotor faces), true north. | yes |
| `Wtc_TDI_ActLimit_Power` | Active power limit, kW. 2,300 or more when the turbine is not limited. | yes |
| `Wtc_TDI_PowerRed_PowerRed` | 1 while the turbine is reducing power, else 0. | yes |
| `Wtc_TDI_GenState_GenState` | Generator state; `LargeOnGrd` in normal production. Not used by the scorer. | yes |
| `computed_core_post_processed_core_wake_steering_offset_degrees` | Wake steering yaw offset the controller computes. Positive is counter-clockwise seen from above. Zero when it wants no steering. It is computed on toggle-off minutes too, but only acted on when the toggle is on. | no |
| `computed_driver_post_processed_yaw_target_degrees` | Yaw target including steering. In the turbine's own north frame, not true north. | no |
| `computed_driver_post_processed_toggle_state` | 1 when the DY controller is in charge, 0 when it is not. | no |

**Yaw misalignment** here means yaw direction minus consensus wind direction, wrapped to ±180°. It is positive
when the rotor is turned clockwise (seen from above) from the wind. In the datapack the steering offset tag has
the opposite sign convention, so a commanded offset of +10° shows up as a misalignment of about −10°, and the yaw
direction lags the command and moves in steps. The measured misalignment is what the turbine actually did, and it
is all you have on scoring days. `hot_floris.make_minute_df` computes a per-minute misalignment this way.

### The ZXTM LiDAR on T07

The ZXTM is a continuous-wave LiDAR on T07's nacelle. It scans a cone of 15° half-angle around T07's rotor axis
and turns with the nacelle, so where it looks depends on T07's yaw direction, which you do not have on scoring
days. Geometry, as confirmed by ZX Lidars and coded in `zxtm_probes.py`:

- The cone axis is horizontal (the instrument corrects for tilt and roll) and starts at the optical head, 3.2 m
  above hub height and 0.8 m behind the rotor plane.
- `Range (m)` is distance along the cone axis. At the 126 m range the scan circle has a radius of 33.8 m and is
  125.2 m in front of the rotor, which is 264 m (3.2 D) downwind of T03 when T07 faces it.
- Each row gives a `Left` and a `Right` LOS speed at five heights above ground ("rotor segment heights"). Left and
  right are as seen from the nacelle looking upwind. The values to predict are at **range 126 m, segment height
  59.0 m** (hub height): datapack columns `Left LOS Speed (m/s) at Rotor Segment Height 59.0m` and the `Right`
  equivalent, in rows with `Range (m)` equal to 126. The beams there are about 33 m either side of the axis.
- An LOS speed is the wind speed component along the beam, about 15° off the axis, and is reported as a
  magnitude: never negative.
- The minute mean is the mean of the high-resolution (`Wind_` file) values in the minute, bad values removed
  (`zxtm_probes.zxtm_minute_means`).

## What to submit

One CSV with the same rows as [`scoring/submission_template.csv`](scoring/submission_template.csv): one row per
minute of every period, 3,450 rows.

| Column | Content |
|---|---|
| `timestamp_utc` | Start of the minute, as in the template (`2026-07-18T21:00:00Z`). Do not change. |
| `pair` | The turbine pair, as in the template. Do not change. |
| `downstream_power_kW` | Your prediction of the downstream turbine's mean power over that minute. |
| `zxtm_126m_left_los_ms` | T03 → T07 rows: the ZXTM Left LOS speed, range 126 m, height 59.0 m, minute mean. |
| `zxtm_126m_right_los_ms` | T03 → T07 rows: the same for the Right beam. |

A submission is rejected if a column is missing; a timestamp is missing, duplicated or extra; a `pair` differs
from the template; a power is blank or not finite; or, in a T03 → T07 row, an LOS speed is blank, not finite or
negative. The two LOS columns are ignored in the other pairs' rows and may be left blank there. Extra columns are
ignored. To check a file before sending it:

```python
from pathlib import Path
from scripts.wedowind_wake_model_2026.scoring import read_submission

read_submission(Path("submission.csv"))  # raises ValueError with the reason if it is not valid
```

## How it is scored

The scorer is [`scoring.py`](scoring.py); the measured values it compares against are not published.

You predict every minute, but only some are scored. A minute is scored for power when all of these hold:

- the upstream turbine's consensus wind direction (minute circular mean) is inside the pair's window:
  T03 → T07 [206°, 222°), T11 → T14 [228°, 256°), T07 → T08 [260°, 295°), T07 → T04 [312°, 342°);
- the upstream turbine's nacelle wind speed (minute mean) is in [6, 12) m/s;
- the upstream turbine's mean power is above 150 kW;
- both turbines operate normally for the whole minute: power above zero every second and not power limited.
  Power limited means `PowerRed_PowerRed` set, or `ActLimit_Power` below 2,300 kW. For T11, whose `ActLimit_Power`
  tracks its power on the scoring days, it instead means `ActLimit_Power` less than 50 kW above its power.

A T03 → T07 minute is scored for the LiDAR as well when both measured LOS values exist. You can work out the
upstream turbine's part of this from your inputs. The downstream turbine's part and the LiDAR's availability you
cannot, so the exact set of scored minutes is not published: about three quarters of all minutes are scored for
power and about half of the T03 → T07 minutes for the LiDAR.

For each scored minute:

- **power error** = (predicted downstream power − measured downstream power) / measured upstream power;
- **L−R** = 100 × (L − R) / ((L + R) / 2), the left-right difference of the two LOS speeds in percent. It is
  computed from your two predictions and from the two measurements; the **L−R error** is the difference, in
  percentage points. Only this asymmetry is scored, not the level of the LOS speeds: it is what moves when the
  wake moves sideways.

There are three power terms, one per upstream turbine (T03, T11, and T07 with its two pairs pooled), each the
root mean square of the power error over that turbine's scored minutes, and one L−R term:

```
score = 0.25 × (RMSE_T03 + RMSE_T11 + RMSE_T07) / 0.4  +  0.25 × RMSE_L−R / 30
```

Lower is better, and a perfect submission scores 0. The divisors 0.4 and 30 are round numbers chosen by hand, near
the per-minute errors of a simple empirical model without steering on toggle-on T03 → T07 minutes in the datapack.
They only put the four terms on a similar scale; a score of about 1 is a rough "no better than ignoring steering".
Per-minute data is noisy and no model gets near 0.

## Things to know about the data

- **T11's direction on 2026-10-15.** T11's north calibration was changed between the summer periods and
  2026-10-15. Compared with its neighbours, T11's consensus wind direction reads about 2–3° further clockwise on
  2026-10-15 than in the summer periods. The scored-minute window uses T11's own direction throughout, but a model
  that turns T11's direction into a wake position may be off by that much on that day.
- **T11 is often power limited** on its scoring days, which is why it has its own rule above. Its limited minutes
  are not scored, but you still have to predict them.
- **Northing in general.** Tags ending `_true_degrees` have the controller's northing correction applied; turbine
  frame tags (`YawPos_Value`, `AcWindDr_Value`, the yaw target) do not. A correction can be a few degrees out and
  can change over time, and a few degrees matter at 5 D spacing.
- **Wake centre against bearing.** Where the data put the centre of a wake, in the upstream turbine's consensus
  direction, can be a few degrees from the bearing between the two turbines. Northing errors and the flow turning
  over the terrain both contribute, so check it in the datapack rather than trusting the geometry.
- **Control may differ from the datapack.** Steering may or may not be on in a scoring period, and its
  calibration may differ from the open data, so a model that only reproduces the datapack's toggle-on behaviour
  may not carry over. A model of how the wake responds to where the upstream turbine's nacelle points should.
- **Season.** The datapack fastlog covers January to April; the scoring periods are July to November, and several
  are at night.

## Worked example: FLORIS

[`example_submission.py`](example_submission.py) writes a valid submission from the public
[FLORIS](https://github.com/NREL/floris) model in this folder. It is a starting point and a format check, not a
strong entry. Per minute it takes the free stream from the upstream turbine (nacelle wind speed times a fitted
factor `k`, consensus wind direction, and a turbulence intensity from the 1 Hz wind speed), gives the upstream
turbine its measured yaw misalignment and reads off the downstream turbine's power. For T03 → T07 it samples the
modelled flow at the two ZXTM beam positions, assuming T07 faces T03's wind direction. Wake parameters are FLORIS
defaults.

```commandline
uv sync --group wedowind
uv run python -m scripts.wedowind_wake_model_2026.example_submission --fastlog-dir "<scoring inputs>/turbine_fastlog/Filestore"
```

Run it from the repository root. Add any other packages your own method needs with `uv pip install`.

It writes `submission.csv` to the output directory (`HOT_OPEN_OUTPUT_DIR`, see the top-level README) and takes a
few minutes. `--no-steering` writes the same model with every turbine at zero yaw.

The other scripts show the model against measurements in the datapack period, on the night of 2026-02-20:

- [`hot_floris.py`](hot_floris.py): the shared model, inputs and filters.
- [`toggle_off_t07.py`](toggle_off_t07.py): fits `k` on T03 and predicts T07 power on toggle-off minutes.
- [`toggle_on_t07.py`](toggle_on_t07.py): the same with T03's measured yaw on toggle-on minutes.
- [`zxtm_probes_t07.py`](zxtm_probes_t07.py): modelled against measured ZXTM LOS speeds.
