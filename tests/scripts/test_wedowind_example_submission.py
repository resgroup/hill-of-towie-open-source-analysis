from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("floris")

from scripts.wedowind_wake_model_2026 import example_submission
from scripts.wedowind_wake_model_2026.example_submission import period_minute_df, write_submission
from scripts.wedowind_wake_model_2026.hot_floris import (
    ACT_LIMIT_COL,
    FREE_STREAM_COLS,
    GEN_STATE_COL,
    POWER_COL,
    POWER_RED_COL,
    TI_WINDOW,
    WD_COL,
    WS_COL,
    YAW_POS_COL,
)
from scripts.wedowind_wake_model_2026.scoring import (
    DOWNSTREAM_POWER_COL,
    LEFT_LOS_COL,
    PAIRS_BY_NAME,
    RIGHT_LOS_COL,
    read_submission,
    submission_template,
)

PAIR = PAIRS_BY_NAME["T03->T07"]
START = pd.Timestamp("2026-09-01 02:00", tz="UTC")
END_EXCL = START + pd.Timedelta(minutes=6)


def _t03_fl_df() -> pd.DataFrame:
    """1 Hz T03 fastlog, steady in the T03 -> T07 window and yawed 10 deg clockwise of the wind."""
    index = pd.date_range(START - TI_WINDOW, END_EXCL + TI_WINDOW, freq="1s", inclusive="left").rename("timestamp")
    d = pd.DataFrame(
        {
            POWER_COL: 800.0,
            WS_COL: np.tile([7.8, 8.2], len(index) // 2),
            GEN_STATE_COL: "LargeOnGrd",
            ACT_LIMIT_COL: 2300.0,
            POWER_RED_COL: 0.0,
            WD_COL: 210.0,
            YAW_POS_COL: 220.0,
        },
        index=index,
    )
    return pd.concat({"T03": d}, axis=1, names=["device_id", "tag"])


def test_period_minute_df_fills_gaps_and_keeps_them_out_of_the_k_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submission needs every minute, so gaps are filled, but filled or unscored minutes must not fit k."""
    fl_df = _t03_fl_df()

    def minute(n: int) -> slice:
        return slice(START + pd.Timedelta(minutes=n), START + pd.Timedelta(minutes=n, seconds=59))

    fl_df.loc[minute(1), ("T03", YAW_POS_COL)] = np.nan
    fl_df.loc[minute(2), ("T03", WS_COL)] = np.nan
    fl_df.loc[minute(3), ("T03", POWER_RED_COL)] = 1.0
    fl_df.loc[minute(4), ("T03", POWER_COL)] = 100.0
    monkeypatch.setattr(example_submission, "load_period_fl_df", lambda *_, **__: fl_df)

    minute_df = period_minute_df(PAIR, start=START, end_excl=END_EXCL, fastlog_dirs=[])

    assert minute_df.index.equals(pd.date_range(START, END_EXCL, freq="1min", inclusive="left"))
    assert minute_df["fit_k"].tolist() == [True, False, False, False, False, True]
    assert minute_df[FREE_STREAM_COLS].notna().all(axis=None)
    assert minute_df["T03_yaw_misalignment"].tolist() == pytest.approx([10, 0, 10, 10, 10, 10])


def test_write_submission_writes_a_file_the_scorer_reads(tmp_path: Path) -> None:
    """The written CSV must pass the scorer's own reader."""
    submission = submission_template()
    submission[[DOWNSTREAM_POWER_COL, LEFT_LOS_COL, RIGHT_LOS_COL]] = [500.0, 6.0, 7.0]
    path = tmp_path / "submission.csv"

    write_submission(submission, path)

    assert read_submission(path)[DOWNSTREAM_POWER_COL].eq(500.0).all()


def test_write_submission_rejects_an_incomplete_submission(tmp_path: Path) -> None:
    """A submission without LOS speeds in the LiDAR pair's minutes must not be written."""
    submission = submission_template()
    submission[DOWNSTREAM_POWER_COL] = 500.0

    with pytest.raises(ValueError, match="missing or non-finite"):
        write_submission(submission, tmp_path / "submission.csv")
