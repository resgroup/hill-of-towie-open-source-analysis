"""Build tiny synthetic Hill of Towie 10-min SCADA year zips, laid out like the Zenodo ones.

Each year zip holds one CSV per ``(table, month)`` named ``{table}_{year}_{month:02d}.csv`` with an
end-format ``TimeStamp`` index, a ``StationId`` column and the raw ``wtc_*`` field columns.
Values are a deterministic function of turbine number and timestamp so tests can assert on them.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from hot_open.scada_helpers import hill_of_towie_fields

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SERIAL_OFFSET = 2304509
ALL_WTG_NUMBERS = list(range(1, 22))


def field_value(*, field_name: str, wtg_number: int, start_ts: pd.Timestamp) -> float:
    """Return the value written for a field, turbine and start-format timestamp."""
    minutes = (start_ts - pd.Timestamp("2000-01-01", tz="UTC")).total_seconds() / 60
    field_idx = [f.field_name for f in hill_of_towie_fields].index(field_name)
    if field_name == "wtc_ScReToOp_timeon":
        return 600.0
    return float(wtg_number * 1000 + field_idx * 10 + (minutes / 10) % 7)


def write_year_zips(
    data_dir: Path,
    *,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    wtg_numbers: Sequence[int] = ALL_WTG_NUMBERS,
    index_name: str = "TimeStamp",
) -> None:
    """Write one zip per year covering ``[start_dt, end_dt_excl)`` in start format."""
    start_stamps = pd.date_range(start_dt, end_dt_excl, freq="10min", inclusive="left")
    tables: dict[str, list[str]] = defaultdict(list)
    for f in hill_of_towie_fields:
        tables[f.table_name].append(f.field_name)
    by_year_month: dict[tuple[int, int], list[pd.Timestamp]] = defaultdict(list)
    for ts in start_stamps:
        by_year_month[(ts.year, ts.month)].append(ts)
    years = sorted({y for y, _ in by_year_month})
    for year in years:
        with zipfile.ZipFile(data_dir / f"{year}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for (y, month), stamps in by_year_month.items():
                if y != year:
                    continue
                for table, field_names in tables.items():
                    rows = [
                        {
                            index_name: (ts + pd.Timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                            "StationId": wtg + SERIAL_OFFSET,
                            **{fn: field_value(field_name=fn, wtg_number=wtg, start_ts=ts) for fn in field_names},
                        }
                        for ts in stamps
                        for wtg in wtg_numbers
                    ]
                    zf.writestr(f"{table}_{year}_{month:02d}.csv", pd.DataFrame(rows).to_csv(index=False))


def write_shutdown_duration_zip(
    data_dir: Path,
    *,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    wtg_numbers: Sequence[int] = ALL_WTG_NUMBERS,
) -> pd.DataFrame:
    """Write ``Hill_of_Towie_ShutdownDuration.zip`` and return the frame written."""
    start_stamps = pd.date_range(start_dt, end_dt_excl, freq="10min", inclusive="left")
    df = pd.DataFrame(
        [
            {
                "TimeStamp_StartFormat": ts.isoformat(sep=" "),
                "TurbineName": f"T{wtg:02d}",
                "ShutdownDuration": (wtg * 7 + i) % 600,
            }
            for i, ts in enumerate(start_stamps)
            for wtg in wtg_numbers
        ]
    )
    with zipfile.ZipFile(data_dir / "Hill_of_Towie_ShutdownDuration.zip", "w") as zf:
        zf.writestr("Hill_of_Towie_ShutdownDuration.csv", df.to_csv(index=False))
    return df
