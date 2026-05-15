from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hot_open.settings import get_out_dir

XLSX_PATH = Path(
    r"C:\Users\aclerc\OneDrive - RES Group\Controls role\Hill of Towie"
    r"\Dynamic Yaw validation 2026\HOT DY validation workings.xlsx"
)
SHEET = "Stddevdir"
HOT_VALUE = 6.759831

if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)

    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET)
    mask = (df["MeanStdDevDir"] > 2) & (df["MeanStdDevDir"] < 14)
    values = df.loc[mask, "MeanStdDevDir"]

    bins = np.arange(2, 16, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, edgecolor="black")
    ax.set_xlabel("MeanStdDevDir [deg]")
    ax.set_ylabel("wind farms")
    ax.set_title("Distribution of MeanStdDevDir across wind farms")
    ax.set_xticks(bins)
    ax.grid(visible=True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    median_value = values.median()
    ax.axvline(median_value, color="black", linestyle="--", linewidth=1.5, label=f"median ({median_value:.2f})")

    ax.axvline(HOT_VALUE, color="red", linestyle="--", linewidth=1.5, label=f"Hill of Towie ({HOT_VALUE:.2f})")
    ymax = ax.get_ylim()[1]
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "stddevdir_histogram.png", dpi=150)
    plt.show()
