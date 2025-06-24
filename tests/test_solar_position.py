import datetime as dt
import math

import pandas as pd
import pytest

from hot_open.solar_position import SolarPositionCalculator


@pytest.fixture
def denver_calculator() -> SolarPositionCalculator:
    return SolarPositionCalculator(latitude=39.7392, longitude=-104.9903)


def test_initialization(denver_calculator: SolarPositionCalculator) -> None:
    calc = denver_calculator
    assert isinstance(calc, SolarPositionCalculator)
    assert calc.observer.lat == 39.7392 * math.pi / 180
    assert calc.observer.lon == -104.9903 * math.pi / 180


def test_single_calculation(denver_calculator: SolarPositionCalculator) -> None:
    # sunrise and sunset times looked up from https://www.timeanddate.com/sun/usa/denver
    # and converted to UTC

    # sunrise
    alt, az = denver_calculator.get_alt_and_az(
        dt.datetime(year=2025, month=5, day=19, hour=11, minute=42, second=0, tzinfo=dt.UTC)
    )
    assert alt == pytest.approx(0, abs=0.2)
    assert az == pytest.approx(63, abs=0.1)

    # sunset
    alt, az = denver_calculator.get_alt_and_az(
        dt.datetime(year=2025, month=5, day=20, hour=2, minute=11, second=0, tzinfo=dt.UTC)
    )
    assert alt == pytest.approx(0, abs=0.2)
    assert az == pytest.approx(297, abs=0.1)


def test_single_calculation_pandas_timestamp(denver_calculator: SolarPositionCalculator) -> None:
    # sunrise and sunset times looked up from https://www.timeanddate.com/sun/usa/denver
    # and converted to UTC

    # sunrise
    alt, az = denver_calculator.get_alt_and_az(
        pd.Timestamp(dt.datetime(year=2025, month=5, day=19, hour=11, minute=42, second=0, tzinfo=dt.UTC))
    )
    assert alt == pytest.approx(0, abs=0.2)
    assert az == pytest.approx(63, abs=0.1)

    # sunset
    alt, az = denver_calculator.get_alt_and_az(
        pd.Timestamp(dt.datetime(year=2025, month=5, day=20, hour=2, minute=11, second=0, tzinfo=dt.UTC))
    )
    assert alt == pytest.approx(0, abs=0.2)
    assert az == pytest.approx(297, abs=0.1)


def test_batch_calculation_pandas_timestamp(denver_calculator: SolarPositionCalculator) -> None:
    # sunrise and sunset times looked up from https://www.timeanddate.com/sun/usa/denver
    # and converted to UTC
    timestamps = [
        pd.Timestamp(dt.datetime(year=2025, month=5, day=19, hour=11, minute=42, second=0, tzinfo=dt.UTC)),
        pd.Timestamp(dt.datetime(year=2025, month=5, day=20, hour=2, minute=11, second=0, tzinfo=dt.UTC)),
    ]
    alt, az = denver_calculator.get_alt_and_az_batch(timestamps)

    assert alt[0] == pytest.approx(0, abs=0.2)
    assert az[0] == pytest.approx(63, abs=0.1)
    assert alt[1] == pytest.approx(0, abs=0.2)
    assert az[1] == pytest.approx(297, abs=0.1)


@pytest.mark.parametrize(
    "timestamp",
    [
        dt.datetime(
            year=2025,
            month=5,
            day=19,
            hour=6,
            minute=42,
            second=0,
            tzinfo=dt.timezone(dt.timedelta(hours=-5), name="EST"),
        ),
        dt.datetime(year=2025, month=5, day=19, hour=11, minute=42, second=0),  # noqa:DTZ001
        pd.Timestamp(dt.datetime(year=2025, month=5, day=19, hour=6, minute=42, second=0), tz="America/New_York"),  # noqa:DTZ001
    ],
    ids=["non_utc_timestamp", "naive_timestamp", "non_utc_pd_timestamp"],
)
def test_non_utc_timestamp_raises_error(timestamp: dt.datetime, denver_calculator: SolarPositionCalculator) -> None:
    with pytest.raises(ValueError, match="Timestamp must be timezone-aware and in UTC."):
        denver_calculator.get_alt_and_az(timestamp)
