import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from hot_open.circular_math import circdiff_degrees, circmedian_degrees

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
