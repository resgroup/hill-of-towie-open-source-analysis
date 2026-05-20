import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from hot_open.circular_math import (
    circ_mean_dataframe_columns,
    circ_mean_resample_degrees,
    circdiff_degrees,
    circmedian_degrees,
)

test_circ_diff_data = [
    (0, 0, 0),
    (2, 1, 1),
    (359, 1, -2),
    (90, 270, -180),
    (90, 270.1, 179.9),
    ([1, 90, 90], [-1, 270, 270.1], [2, -180, 179.9]),
    (pd.Series([1, 90, 90]), pd.Series([-1, 270, 270.1]), pd.Series([2, -180, 179.9])),
    (pd.Series([1, 359.1, 2.1]), 1, pd.Series([0, -1.9, 1.1])),
    (1, pd.Series([1, 359.1, 2.1]), pd.Series([0, 1.9, -1.1])),
]


@pytest.mark.parametrize(("angle1", "angle2", "expected"), test_circ_diff_data)
def test_circdiff_degrees(
    angle1: float | npt.NDArray, angle2: float | npt.NDArray, expected: float | npt.NDArray
) -> None:
    """Test the circdiff_degrees function."""
    if isinstance(expected, pd.Series):
        assert_series_equal(circdiff_degrees(angle1, angle2), (expected))
        assert_series_equal(circdiff_degrees(angle1 + 360, angle2), (expected))
        assert_series_equal(circdiff_degrees(angle1 - 360, angle2), (expected))
    else:
        assert circdiff_degrees(angle1, angle2) == pytest.approx(expected)


test_circmedian_degrees_data = [
    ([0], 0),
    ([0, 1], 0.5),
    ([359, 2], 0.5),
    ([0, 1, 1], 1),
    ([359, 359, 1], 359),
    ([358, 359, 1], 359),
    ([359, 0, 10, 20, 30, 31], 15),
    ([np.nan, 358, 359, 1], 359),
    ([np.nan, 359, 0, 10, 20, 30, 31], 15),
]


@pytest.mark.parametrize(("angles", "expected"), test_circmedian_degrees_data)
def test_circmedian_degrees(angles: npt.NDArray, expected: float) -> None:
    """Test the circmedian_degrees function."""
    assert circmedian_degrees(angles) == pytest.approx(expected)
    assert circmedian_degrees([x + 360 for x in angles]) == pytest.approx(expected)
    assert circmedian_degrees([x - 360 for x in angles]) == pytest.approx(expected)


def test_circmedian_degrees_empty_raises() -> None:
    """Empty input raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        circmedian_degrees([])


def test_circmedian_degrees_all_nan_returns_nan() -> None:
    """All-NaN input returns NaN rather than raising."""
    assert np.isnan(circmedian_degrees([np.nan, np.nan]))


def test_circ_mean_resample_degrees_basic() -> None:
    """Resample circular data; values that straddle 0/360 should average correctly."""
    index = pd.date_range("2024-01-01", periods=6, freq="10min")
    df = pd.DataFrame(
        {
            "a": [0.0, 10.0, 20.0, 100.0, 110.0, 120.0],
            "b": [350.0, 10.0, 0.0, 180.0, 180.0, 180.0],
        },
        index=index,
    )
    result = circ_mean_resample_degrees(df, resample_timedelta=pd.Timedelta("30min"))
    assert list(result.index) == [index[0], index[3]]
    assert result["a"].iloc[0] == pytest.approx(10.0)
    assert result["a"].iloc[1] == pytest.approx(110.0)
    # Mean of 350, 10, 0 wraps across zero — expected ~0 (i.e. 360)
    assert result["b"].iloc[0] % 360 == pytest.approx(0.0, abs=1e-6)
    assert result["b"].iloc[1] == pytest.approx(180.0)


def test_circ_mean_resample_degrees_requires_datetime_index() -> None:
    """A non-DatetimeIndex DataFrame raises TypeError."""
    df = pd.DataFrame({"a": [0.0, 90.0]}, index=[0, 1])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        circ_mean_resample_degrees(df, resample_timedelta=pd.Timedelta("10min"))


def test_circ_mean_resample_degrees_warns_on_out_of_range(caplog: pytest.LogCaptureFixture) -> None:
    """Values outside [0, 360] emit a warning but do not raise."""
    index = pd.date_range("2024-01-01", periods=2, freq="10min")
    df = pd.DataFrame({"a": [-10.0, 10.0]}, index=index)
    with caplog.at_level("WARNING", logger="hot_open.circular_math"):
        circ_mean_resample_degrees(df, resample_timedelta=pd.Timedelta("10min"))
    assert any("outside 0 to 360" in r.message for r in caplog.records)


def test_circ_mean_resample_degrees_warns_on_range_above_360(caplog: pytest.LogCaptureFixture) -> None:
    """A data range greater than 360 emits a warning."""
    index = pd.date_range("2024-01-01", periods=2, freq="10min")
    df = pd.DataFrame({"a": [0.0, 720.0]}, index=index)
    with caplog.at_level("WARNING", logger="hot_open.circular_math"):
        circ_mean_resample_degrees(df, resample_timedelta=pd.Timedelta("10min"))
    assert any("range larger than 360" in r.message for r in caplog.records)


def test_circ_mean_dataframe_columns_basic() -> None:
    """Row-wise circular mean across columns."""
    df = pd.DataFrame(
        {
            "a": [359.0, 90.0, 350.0],
            "b": [11.0, 90.0, 10.0],
        }
    )
    result = circ_mean_dataframe_columns(df)
    assert result.iloc[0] == pytest.approx(5.0)
    assert result.iloc[1] == pytest.approx(90.0)
    # 350 and 10 average across the 0/360 wrap to 0
    assert result.iloc[2] % 360 == pytest.approx(0.0, abs=1e-6)


def test_circ_mean_dataframe_columns_warns_on_out_of_range(caplog: pytest.LogCaptureFixture) -> None:
    """Values outside [0, 360) emit a warning but do not raise."""
    df = pd.DataFrame({"a": [-1.0, 10.0], "b": [10.0, 20.0]})
    with caplog.at_level("WARNING", logger="hot_open.circular_math"):
        circ_mean_dataframe_columns(df)
    assert any("between 0 and 360" in r.message for r in caplog.records)


def test_circ_mean_dataframe_columns_raises_on_range_above_360() -> None:
    """A column with a range greater than 360 raises ValueError."""
    df = pd.DataFrame({"a": [0.0, 720.0], "b": [10.0, 20.0]})
    with pytest.raises(ValueError, match="range larger than 360"):
        circ_mean_dataframe_columns(df)
