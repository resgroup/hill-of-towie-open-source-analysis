"""Circular math functions missing from numpy/scipy."""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import circmean

logger = logging.getLogger(__name__)


def circdiff_degrees(
    angle1: float | npt.NDArray | list | pd.Series, angle2: float | npt.NDArray | list | pd.Series
) -> float | npt.NDArray | pd.Series:
    """Calculate the circular difference between two angles.

    :param angle1: First angle in degrees.
    :param angle2: Second angle in degrees.
    :return: Circular difference between the two angles in degrees
    """
    # Convert list to numpy array
    if isinstance(angle1, list):
        angle1 = np.array(angle1)
    if isinstance(angle2, list):
        angle2 = np.array(angle2)

    return np.mod(angle1 - angle2 + 180, 360) - 180


def circ_mean_resample_degrees(df: pd.DataFrame, *, resample_timedelta: pd.Timedelta) -> pd.DataFrame:
    """Apply resample and ciruclar mean to a DataFrame efficiently.

    Args:
    ----
        df: DataFrame where each column contains circular data in degrees from 0 inclusive to 360 exclusive.
        resample_timedelta: Timebase to resample the data to.

    Returns:
    -------
        Resampled DataFrame with circular mean of each column.

    """
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "DataFrame must have a DatetimeIndex."
        raise TypeError(msg)

    for col in df.columns:
        if (data_range := df[col].max() - df[col].min()) > 360:  # noqa:PLR2004
            msg = f"Data in circular column {col} has a range larger than 360: {data_range}."
            logger.warning(msg)
        elif (df[col].min() < 0) or (df[col].max() > 360):  # noqa:PLR2004
            msg = (
                f"Data in circular column {col} has values outside 0 to 360: "
                f"min = {df[col].min()}, max = {df[col].max()}."
            )
            logger.warning(msg)

    rad_df = np.deg2rad(df)
    sin_df = pd.DataFrame(np.sin(rad_df), index=df.index, columns=df.columns)
    cos_df = pd.DataFrame(np.cos(rad_df), index=df.index, columns=df.columns)
    sin_df_resampled = sin_df.resample(resample_timedelta).mean()
    cos_df_resampled = cos_df.resample(resample_timedelta).mean()
    result = (np.rad2deg(np.arctan2(sin_df_resampled, cos_df_resampled)) + 360) % 360
    return pd.DataFrame(result, index=sin_df_resampled.index, columns=df.columns)


def circ_mean_dataframe_columns(df: pd.DataFrame) -> pd.Series:
    """Apply ciruclar mean to DataFrame columns efficiently.

    Args:
    ----
        df: DataFrame where each column contains circular data in degrees from 0 inclusive to 360 exclusive.
        resample_timedelta: Timebase to resample the data to.

    Returns:
    -------
        Resampled DataFrame with circular mean of each column.

    """
    # warn if data outside 0 to 360
    if any(df.min() < 0) or any(df.max() >= 360):  # noqa:PLR2004
        msg = "Data is expected to be between 0 and 360 degrees."
        logger.warning(msg)

    # check that each column has data where max-min is no more than
    for col in df.columns:
        if (data_range := df[col].max() - df[col].min()) > 360:  # noqa:PLR2004
            msg = f"Data in column {col} has a range larger than 360 degrees: {data_range}."
            raise ValueError(msg)

    rad_df = np.deg2rad(df)
    sin_df = np.sin(rad_df)
    cos_df = np.cos(rad_df)
    mean_sin_df = sin_df.mean(axis=1)
    mean_cos_df = cos_df.mean(axis=1)
    return (np.rad2deg(np.arctan2(mean_sin_df, mean_cos_df)) + 360) % 360


def circmedian_degrees(angles: list[float] | npt.NDArray | pd.Series) -> float:
    """Calculate the circular median of angles in degrees.

    The circular median is defined as the angle in provided angles that minimizes the sum of circular
    distances to all other provided angles. For a set of angles, this is computed
    by converting to the unit circle, finding the angle that minimizes the great circle
    distances, and converting back to degrees.

    If two angles are tied their circular mean is returned.

    Args:
    ----
        angles: Array-like object containing angles in degrees

    Returns:
    -------
        float: Circular median in degrees, normalized to [0, 360)

    """
    # Convert to numpy array and handle empty input
    angles_arr = np.asarray(angles)
    if len(angles_arr) == 0:
        msg = "Input array must not be empty"
        raise ValueError(msg)

    # Remove NaN values
    valid_angles = angles_arr[~np.isnan(angles_arr)]

    if len(valid_angles) == 0:
        return np.nan

    # Find the angle that minimizes the sum of absolute circular differences
    min_sum_diff = float("inf")
    median_angle = np.nan
    candidates_which_are_tied = []

    for candidate in valid_angles:
        # Calculate absolute circular differences to all other angles
        diffs = np.abs(circdiff_degrees(valid_angles, candidate))
        sum_diff = np.sum(diffs)

        if sum_diff < min_sum_diff:
            min_sum_diff = sum_diff
            median_angle = candidate
            candidates_which_are_tied = [candidate]
        elif sum_diff == min_sum_diff:
            candidates_which_are_tied.append(candidate)

    return float(
        (median_angle + 360) % 360
        if len(candidates_which_are_tied) == 1
        else circmean(candidates_which_are_tied, low=0, high=360)
    )
