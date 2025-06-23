import datetime as dt
from collections.abc import Sequence

import ephem
import numpy as np
import numpy.typing as npt


class SolarPositionCalculator:
    def __init__(self, *, latitude: float, longitude: float):
        self.observer = ephem.Observer()
        self.observer.lat = str(latitude)
        self.observer.lon = str(longitude)
        self.sun = ephem.Sun()

    def _validate_utc(self, timestamp: dt.datetime) -> None:
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) != dt.timedelta(0):
            msg = "Timestamp must be timezone-aware and in UTC."
            raise ValueError(msg)

    def get_alt_and_az(self, timestamp: dt.datetime) -> tuple[float, float]:
        self._validate_utc(timestamp)
        self.observer.date = timestamp.strftime("%Y/%m/%d %H:%M:%S")
        self.sun.compute(self.observer)
        return np.degrees(self.sun.alt), np.degrees(self.sun.az)

    def get_alt_and_az_batch(self, timestamps: Sequence[dt.datetime]) -> tuple[npt.NDArray, npt.NDArray]:
        alt_values_radians = []
        az_values_radians = []

        for timestamp in timestamps:
            self._validate_utc(timestamp)
            self.observer.date = timestamp.strftime("%Y/%m/%d %H:%M:%S")
            self.sun.compute(self.observer)
            alt_values_radians.append(self.sun.alt)
            az_values_radians.append(self.sun.az)

        return np.degrees(alt_values_radians), np.degrees(az_values_radians)
