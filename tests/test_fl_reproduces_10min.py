"""Fastlog resampled to 600s must reproduce the turbine's own 10-minute SCADA statistics.

This is the strongest check available on the resampler: every other test pins its output against
values this repo produced itself, whereas these compare against aggregates the controller computed
independently, from the same physical signal, at the time. It covers the forward fill, the duration
weighting, the circular mean, ``std_tags`` and the min/max path in one go.

Two turbine-days, because neither can do the job alone:

* **2026-03-17** carries all four compared signals and a mixture of operating states -- 25 periods
  not generating, 75 in region 2 and 43 at rated -- but its yaw never crosses the 0/360 wrap, so it
  cannot test circular averaging at all.
* **2026-04-25** has 23 wrap-crossing periods. It carries only the direction and one busy tag to
  drive the resample, which keeps the committed fixture small.

The tolerances below are frozen from measurement, not chosen: each is the worst case observed over
the day, rounded up. They are tight enough that a regression in any part of the aggregation path
would break them.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import pytest

from hot_open.fastlog_helpers import get_fl_resampled
from hot_open.scada_helpers import WPSBackupFileField, load_hot_10min_data

TEST_DATA_DIR = Path(__file__).parent / "test_data"
FL_SAMPLE_DIR = TEST_DATA_DIR / "fl_sample"
DEVICE_ID = "2304510"
WTG_NUMBER = 1
TIMEBASE_S = 600
RATED_KW = 2300.0

# 10-minute field stem -> (table, fastlog tag). wtc_GenRpm_* is deliberately absent: fastlog carries
# main-shaft MainSRpm_Value, a different signal, so a disagreement there would mean nothing. The
# 10-minute tables carry no wind direction, so the vane cannot be cross-checked this way either.
FIELD_PAIRS = {
    "wtc_ActPower": ("tblSCTurGrid", "ActPower_Value"),
    "wtc_AcWindSp": ("tblSCTurbine", "AcWindSp_AcWindSp"),
    "wtc_NacelPos": ("tblSCTurbine", "YawPos_Value"),
    "wtc_PitcPosA": ("tblSCTurbine", "PitcPosA_Value"),
}
CIRCULAR_FIELDS = {"wtc_NacelPos"}
STATS = ("mean", "min", "max", "stddev")
FL_STAT_PREFIX = {"mean": "", "min": "min_", "max": "max_", "stddev": "std_"}

DAY_ONE = pd.Timestamp("2026-03-17", tz="UTC")
DAY_ONE_TAGS = ["ActPower_Value", "AcWindSp_AcWindSp", "YawPos_Value", "PitcPosA_Value"]
DAY_TWO = pd.Timestamp("2026-04-25", tz="UTC")
DAY_TWO_TAGS = ["AcWindSp_AcWindSp", "YawPos_Value"]

# Worst absolute difference observed per field and statistic, rounded up. Power is quantised to 1kW
# in the 10-minute record, which is why its min/max tolerances are whole numbers rather than small
# fractions. Circular min/max are excluded on the wrap-crossing day: the minimum and maximum of an
# angle are not well defined across 0/360 under any convention, so a disagreement there is expected.
DAY_ONE_TOLERANCES = {
    ("wtc_ActPower", "mean"): 0.15,  # measured 0.0931
    ("wtc_ActPower", "min"): 2.0,  # measured 1.0
    ("wtc_ActPower", "max"): 3.0,  # measured 2.0
    ("wtc_ActPower", "stddev"): 0.15,  # measured 0.0888
    ("wtc_AcWindSp", "mean"): 0.01,  # measured 0.0026
    ("wtc_AcWindSp", "min"): 0.01,  # measured 0.0
    ("wtc_AcWindSp", "max"): 0.01,  # measured 0.0
    ("wtc_AcWindSp", "stddev"): 0.01,  # measured 0.0021
    ("wtc_NacelPos", "mean"): 0.02,  # measured 0.0050
    ("wtc_NacelPos", "min"): 0.01,  # measured 0.0
    ("wtc_NacelPos", "max"): 0.01,  # measured 0.0
    ("wtc_NacelPos", "stddev"): 0.05,  # measured 0.0251
    ("wtc_PitcPosA", "mean"): 0.01,  # measured 0.0007
    ("wtc_PitcPosA", "min"): 0.01,  # measured 0.0
    ("wtc_PitcPosA", "max"): 0.01,  # measured 0.0
    ("wtc_PitcPosA", "stddev"): 0.01,  # measured 0.0008
}
DAY_TWO_TOLERANCES = {
    ("wtc_AcWindSp", "mean"): 0.02,  # measured 0.0108
    ("wtc_AcWindSp", "stddev"): 0.02,  # measured 0.0105
    ("wtc_NacelPos", "mean"): 0.02,  # measured 0.0091, and its worst case IS a wrap-crossing period
    ("wtc_NacelPos", "stddev"): 0.05,  # measured 0.0298
}

# One turbine-day of this fixture resamples to a 600s product in about 3.5s on a developer machine
# (958,202 raw rows, roughly 270k rows/s) and reads back from cache in 0.02s. The assertion below is
# a loose ceiling rather than that figure: a shared CI runner is slower for reasons that have
# nothing to do with this code, and a tight bound would fail for those reasons alone. The number to
# read is the one the test prints.
REFERENCE_COLD_SECONDS = 3.5
COLD_SECONDS_CEILING = 120.0


def _custom_fields(tags: list[str]) -> list[WPSBackupFileField]:
    return [
        WPSBackupFileField(alias=f"{stem}_{stat}", field_name=f"{stem}_{stat}", table_name=table)
        for stem, (table, tag) in FIELD_PAIRS.items()
        if tag in tags
        for stat in STATS
    ]


def _circular_difference(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b + 180.0) % 360.0 - 180.0


def _difference(joined: pd.DataFrame, field: str, stat: str) -> pd.Series:
    """Return FL minus SCADA for one field and statistic, wrapped where the field is an angle."""
    scada_col = f"{field}_{stat}"
    fl_col = f"{FL_STAT_PREFIX[stat]}{FIELD_PAIRS[field][1]}"
    pair = joined[[scada_col, fl_col]].dropna()
    if field in CIRCULAR_FIELDS and stat != "stddev":
        return _circular_difference(pair[fl_col], pair[scada_col])
    return pair[fl_col] - pair[scada_col]


def _resample(day: pd.Timestamp, tags: list[str], cache_dir: Path, grid: object = "auto") -> tuple[pd.DataFrame, float]:
    start_time = time.perf_counter()
    fl = get_fl_resampled(
        park_id="HOT",
        device_ids=[DEVICE_ID],
        start_dt=day,
        end_dt_excl=day + pd.Timedelta(days=1),
        timebase_s=TIMEBASE_S,
        tags=tags,
        minmax_tags=tags,
        std_tags=tags,
        subsampling_timebase_ms=grid,
        cache_dir=cache_dir,
        filestore_dir=FL_SAMPLE_DIR,
    )
    elapsed = time.perf_counter() - start_time
    fl.columns = fl.columns.droplevel(0)
    return fl, elapsed


def _scada(day: pd.Timestamp, tags: list[str]) -> pd.DataFrame:
    scada = load_hot_10min_data(
        data_dir=TEST_DATA_DIR,
        wtg_numbers=[WTG_NUMBER],
        start_dt=day,
        end_dt_excl=day + pd.Timedelta(days=1),
        custom_fields=_custom_fields(tags),
    )
    scada.columns = scada.columns.droplevel(0)
    return scada


def _join(scada: pd.DataFrame, fl: pd.DataFrame) -> pd.DataFrame:
    """Join the two products, dropping the day's final period.

    The fixture holds single days, so the last raw sample of each has no successor to close out its
    dwell time and the final window is duration-weighted short. That is a property of a series that
    stops, not a defect -- against the full filestore the next day's first sample closes it -- and
    ``test_only_the_trailing_window_is_edge_affected`` pins that it is the only period involved.
    """
    return scada.join(fl, how="inner").iloc[:-1]


@pytest.fixture(scope="module")
def day_one(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Resample the four-signal, mixed-operating-state day, once for every comparison below."""
    fl, _ = _resample(DAY_ONE, DAY_ONE_TAGS, tmp_path_factory.mktemp("cache_day_one"))
    return _join(_scada(DAY_ONE, DAY_ONE_TAGS), fl)


@pytest.fixture(scope="module")
def day_two(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Resample the wrap-crossing day, the only one that can test circular averaging."""
    fl, _ = _resample(DAY_TWO, DAY_TWO_TAGS, tmp_path_factory.mktemp("cache_day_two"))
    return _join(_scada(DAY_TWO, DAY_TWO_TAGS), fl)


class TestFixtureCoversWhatItClaims:
    """Guards against the comparisons below passing on a day that never exercised anything."""

    def test_day_one_mixes_operating_states(self, day_one: pd.DataFrame) -> None:
        power = day_one["wtc_ActPower_mean"]
        assert int((power < 10).sum()) == 25  # not generating
        assert int(((power >= 10) & (power <= 0.95 * RATED_KW)).sum()) == 75  # region 2
        assert int((power > 0.95 * RATED_KW).sum()) == 43  # rated

    def test_day_two_crosses_the_wrap(self, day_two: pd.DataFrame) -> None:
        assert int(_wrap_crossing(day_two).sum()) == 23

    def test_day_one_never_crosses_the_wrap(self, day_one: pd.DataFrame) -> None:
        # States why two days are needed: this one cannot test circular averaging at all.
        assert int(_wrap_crossing(day_one).sum()) == 0


def _wrap_crossing(joined: pd.DataFrame) -> pd.Series:
    """Periods whose yaw spans the 0/360 discontinuity, identified from the fastlog extremes."""
    spread = joined["max_YawPos_Value"] - joined["min_YawPos_Value"]
    return spread > 180.0


class TestTimestampConvention:
    """The 10-minute record labels the END of each period; load_hot_10min_data shifts it to the start."""

    def test_alignment_is_confirmed_by_the_data(self, day_one: pd.DataFrame) -> None:
        # Checked against the data rather than read off the loader, because a one-period join error
        # is the likeliest way to produce a large uniform disagreement across every field at once --
        # which would look like a resampler bug rather than an alignment one.
        scada_power = day_one["wtc_ActPower_mean"]
        fl_power = day_one["ActPower_Value"]
        at_lag = {lag: scada_power.corr(fl_power.shift(lag)) for lag in (-1, 0, 1)}
        assert at_lag[0] > 0.9999
        assert at_lag[-1] < at_lag[0]
        assert at_lag[1] < at_lag[0]


class TestReproducesScadaStatistics:
    @pytest.mark.parametrize(("field", "stat"), list(DAY_ONE_TOLERANCES))
    def test_day_one(self, day_one: pd.DataFrame, field: str, stat: str) -> None:
        worst = _difference(day_one, field, stat).abs().max()
        assert worst <= DAY_ONE_TOLERANCES[(field, stat)], f"{field} {stat}: worst |difference| {worst}"

    @pytest.mark.parametrize(("field", "stat"), list(DAY_TWO_TOLERANCES))
    def test_day_two(self, day_two: pd.DataFrame, field: str, stat: str) -> None:
        worst = _difference(day_two, field, stat).abs().max()
        assert worst <= DAY_TWO_TOLERANCES[(field, stat)], f"{field} {stat}: worst |difference| {worst}"

    def test_circular_mean_holds_across_the_wrap(self, day_two: pd.DataFrame) -> None:
        # The question this day exists for: is the agreement on the wrap-crossing periods as good as
        # on the ordinary ones? It is -- so both implementations do proper circular averaging, and
        # neither is averaging an angle arithmetically across the discontinuity.
        crossing = _wrap_crossing(day_two)
        difference = _difference(day_two, "wtc_NacelPos", "mean").abs()
        assert difference[crossing].max() <= 0.02
        assert difference[crossing].max() <= difference[~crossing].max() * 3


class TestTheGridMustFollowTheLoggingRate:
    """Evidence for the ``subsampling_timebase_ms="auto"`` rule, and a guard on the tolerances above.

    The 10-minute record is the only independent statement of what these aggregates should be, and
    it says the controller aggregated the full-rate signal rather than a 1Hz sample of it: on a 1s
    sub-grid the extremes are systematically inward, and they are not on a grid that follows the
    logging rate. Without this test the tolerances above could be met by loosening them rather than
    by sampling correctly.
    """

    @pytest.fixture(scope="class")
    def one_second_grid(self, tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
        fl, _ = _resample(DAY_ONE, DAY_ONE_TAGS, tmp_path_factory.mktemp("cache_1s_grid"), grid=1000)
        return _join(_scada(DAY_ONE, DAY_ONE_TAGS), fl)

    def test_a_one_second_grid_clips_the_extremes(self, one_second_grid: pd.DataFrame) -> None:
        # Median, not worst case: the bias is systematic, present in most periods of the day. It is
        # also signed, inward from both ends -- a minimum that reads too high and a maximum that
        # reads too low -- which is what an excursion shorter than the grid looks like when missed.
        assert _difference(one_second_grid, "wtc_ActPower", "min").median() >= 4.0
        assert _difference(one_second_grid, "wtc_ActPower", "max").median() <= -8.0

    def test_the_tuned_grid_does_not(self, day_one: pd.DataFrame) -> None:
        assert _difference(day_one, "wtc_ActPower", "min").median() == 0.0
        assert _difference(day_one, "wtc_ActPower", "max").median() == 0.0

    def test_the_mean_is_far_less_grid_sensitive_than_the_extremes(self, one_second_grid: pd.DataFrame) -> None:
        # Sub-sampling a stationary signal estimates its mean with more noise but no real bias; what
        # it loses is any excursion shorter than the grid. Worth pinning, because it is the opposite
        # of what one would guess, and it is why the rule is about min/max rather than about spread.
        coarse_mean = _difference(one_second_grid, "wtc_ActPower", "mean").median()
        assert abs(coarse_mean) < 0.05
        assert abs(_difference(one_second_grid, "wtc_ActPower", "min").median()) > 4.0


class TestTrailingWindow:
    def test_only_the_trailing_window_is_edge_affected(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        """The final period of a day that has no successor is the one the comparisons exclude.

        Pinned rather than quietly dropped: if a change ever made the edge effect reach further back
        than one window, the tolerances above would keep passing and the cause would be invisible.
        """
        fl, _ = _resample(DAY_ONE, DAY_ONE_TAGS, tmp_path_factory.mktemp("cache_trailing"))
        untrimmed = _scada(DAY_ONE, DAY_ONE_TAGS).join(fl, how="inner")
        difference = _difference(untrimmed, "wtc_PitcPosA", "mean").abs()
        assert difference.idxmax() == untrimmed.index[-1]
        assert difference.iloc[-1] > 10 * difference.iloc[:-1].max()


class TestBenchmark:
    def test_resample_throughput(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Record what a turbine-day costs, cold and warm, and fail only on a gross regression."""
        with caplog.at_level(logging.CRITICAL):  # the timing is the point, not the log volume
            _, cold_seconds = _resample(DAY_ONE, DAY_ONE_TAGS, tmp_path)
            _, warm_seconds = _resample(DAY_ONE, DAY_ONE_TAGS, tmp_path)
        raw_rows = 958_202  # the fixture's four tags for this day
        print(  # noqa: T201  -- the measurement is what this test is for
            f"\n600s resample of one turbine-day ({raw_rows:,} raw rows): "
            f"cold {cold_seconds:.1f}s ({raw_rows / cold_seconds / 1e3:.0f}k rows/s), warm {warm_seconds:.2f}s "
            f"[reference cold {REFERENCE_COLD_SECONDS}s]"
        )
        assert cold_seconds < COLD_SECONDS_CEILING
        assert warm_seconds < cold_seconds  # the per-day cache is doing something
