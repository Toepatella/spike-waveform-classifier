"""Task 0: decide which sorted units count as "good cells".

Run with:  uv run good_cells.py

What it does, for each probe in the session:
  1. Reads each unit's quality metrics, depth on the probe and template similarity.
  2. Labels each unit with its brain area using Juan's region limits.
  3. Applies the good-cell rule below.
  4. Flags (but keeps) units with low presence and possible duplicate units.
  5. Saves a table of every unit, plus figures, to the outputs/ folder.

The good-cell rule is set in the RULE section just below. Change the numbers
there and rerun; everything else follows.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import assign_area, load_analyzer, load_area_limits
from paths import DATA_DIR

SESSION = "2026-07-21"
PROBES = ["PFC", "HPC"]
OUTPUT_DIR = Path("outputs")

# ---------------------------------------------------------------------------
# RULE: what counts as a good cell
# The goal is a clean average waveform for each neuron, so every cutoff is
# about that.
# ---------------------------------------------------------------------------

# Contamination: a unit that fires twice within the refractory period (~1.5 ms)
# probably contains spikes from another neuron, which would blend two shapes
# into one average.
MAX_ISI_VIOLATIONS_RATIO = 0.5

# Enough spikes for a clean average: averaging N spikes cuts noise by sqrt(N),
# so 200 spikes cuts it about 14-fold.
MIN_NUM_SPIKES = 200

# The spike must stand out from background noise, or shape measurements such
# as half-width become unreliable.
MIN_SNR = 3.0

# Presence is NOT used to reject units. Low-presence units in this data are
# mostly slow-firing cells, and rejecting them would bias the dataset toward
# fast-firing (often narrow-spiking) neurons. Units below this are only flagged
# so they can be checked by eye.
FLAG_PRESENCE_BELOW = 0.5

# Two units with near-identical average waveforms may be one neuron that the
# sorter split in two. Pairs above this similarity are flagged, not removed.
FLAG_SIMILARITY_ABOVE = 0.95


def build_unit_table(probe: str) -> pd.DataFrame:
    """One row per unit: area, depth, quality metrics and pass/fail columns."""
    analyzer = load_analyzer(SESSION, probe)
    unit_ids = analyzer.unit_ids

    # Metrics come from Juan's quality_metrics.csv, not the analyzer's saved
    # copy: the saved copy is missing SNR for every HPC unit. The CSV's SNR
    # matches a recomputation from the saved templates and noise levels
    # (peak amplitude / noise on the main channel).
    metrics = pd.read_csv(DATA_DIR / SESSION / probe / "quality_metrics.csv", index_col=0)

    # Depth = the unit's estimated position along the probe, in um from the tip.
    locations = analyzer.get_extension("unit_locations").get_data()
    limits = load_area_limits(SESSION, probe)

    table = pd.DataFrame(
        {
            "probe": probe,
            "unit_id": unit_ids,
            "depth_um": locations[:, 1],
            "area": [assign_area(y, limits) for y in locations[:, 1]],
            "num_spikes": metrics.loc[unit_ids, "num_spikes"].to_numpy(),
            "firing_rate_hz": metrics.loc[unit_ids, "firing_rate"].to_numpy(),
            "isi_violations_ratio": metrics.loc[unit_ids, "isi_violations_ratio"].to_numpy(),
            "snr": metrics.loc[unit_ids, "snr"].to_numpy(),
            "presence_ratio": metrics.loc[unit_ids, "presence_ratio"].to_numpy(),
        }
    )

    # Apply the rule. Each criterion gets its own column so you can see why a
    # unit failed.
    table["pass_isi"] = table["isi_violations_ratio"] < MAX_ISI_VIOLATIONS_RATIO
    table["pass_spikes"] = table["num_spikes"] >= MIN_NUM_SPIKES
    table["pass_snr"] = table["snr"] >= MIN_SNR
    table["good"] = table["pass_isi"] & table["pass_spikes"] & table["pass_snr"]

    # Flags: kept in the data, marked for a visual check.
    table["flag_low_presence"] = table["presence_ratio"] < FLAG_PRESENCE_BELOW
    table["possible_duplicate_of"] = find_possible_duplicates(analyzer, table)
    return table


def find_possible_duplicates(analyzer, table: pd.DataFrame) -> list:
    """For each unit, the unit it most resembles if similarity is very high.

    Only the smaller unit of each pair (fewer spikes) gets the flag, since that
    is the one more likely to be a split-off fragment.
    """
    similarity = analyzer.get_extension("template_similarity").get_data().copy()
    np.fill_diagonal(similarity, 0)
    num_spikes = table["num_spikes"].to_numpy()
    unit_ids = table["unit_id"].to_numpy()

    flags = [None] * len(unit_ids)
    for i, j in zip(*np.where(np.triu(similarity) > FLAG_SIMILARITY_ABOVE)):
        smaller, larger = (i, j) if num_spikes[i] < num_spikes[j] else (j, i)
        flags[smaller] = unit_ids[larger]
    return flags


def plot_metric_histograms(table: pd.DataFrame, probe: str) -> Path:
    """Histograms of each metric with the cutoff drawn, so you can see what it removes."""
    panels = [
        # (column, cutoff line, log scale?, label)
        ("isi_violations_ratio", MAX_ISI_VIOLATIONS_RATIO, True, f"keep < {MAX_ISI_VIOLATIONS_RATIO}"),
        ("num_spikes", MIN_NUM_SPIKES, True, f"keep >= {MIN_NUM_SPIKES}"),
        ("snr", MIN_SNR, False, f"keep >= {MIN_SNR}"),
        ("presence_ratio", FLAG_PRESENCE_BELOW, False, f"flag < {FLAG_PRESENCE_BELOW} (not removed)"),
        ("firing_rate_hz", None, True, "no cutoff"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5))
    for ax, (column, cutoff, log_scale, label) in zip(axes, panels):
        values = table[column].to_numpy(dtype=float)
        if log_scale:
            # Log axes can't show zero, so zeros are drawn at a small value.
            smallest = values[values > 0].min()
            values = np.where(values > 0, values, smallest / 2)
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
            ax.set_xscale("log")
        else:
            bins = 40
        ax.hist(values, bins=bins, color="0.6")
        if cutoff is not None:
            ax.axvline(cutoff, color="tab:red", linestyle="--", label=label)
            ax.legend(fontsize=8)
        ax.set_title(column)
        ax.set_ylabel("units")
    good = int(table["good"].sum())
    fig.suptitle(f"{probe}: {good} of {len(table)} units are good")
    fig.tight_layout()
    path = OUTPUT_DIR / f"task0_metrics_{SESSION}_{probe}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_depths(table: pd.DataFrame, probe: str) -> Path:
    """Each unit's depth on the probe with the area boundaries.

    Useful for checking the area labels with Juan.
    """
    limits = load_area_limits(SESSION, probe)
    fig, ax = plt.subplots(figsize=(5, 7))
    rejected = table[~table["good"]]
    good = table[table["good"]]
    ax.scatter(rejected["firing_rate_hz"], rejected["depth_um"], s=10, color="0.75", label="rejected")
    ax.scatter(good["firing_rate_hz"], good["depth_um"], s=10, color="tab:blue", label="good")
    top_of_channels = table["depth_um"].max()
    for area, top in limits.items():
        if top <= top_of_channels + 200:
            ax.axhline(top, color="tab:red", linestyle="--", linewidth=1)
            ax.text(ax.get_xlim()[1], top, f" top of {area}", va="bottom", ha="right", fontsize=8, color="tab:red")
    ax.set_xscale("log")
    ax.set_xlabel("firing rate (Hz)")
    ax.set_ylabel("depth from probe tip (um)")
    ax.set_title(f"{probe}: unit depths and area limits")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / f"task0_depths_{SESSION}_{probe}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tables = []
    for probe in PROBES:
        print(f"Loading {probe} ...")
        table = build_unit_table(probe)
        tables.append(table)
        print(f"  saved {plot_metric_histograms(table, probe)}")
        print(f"  saved {plot_depths(table, probe)}")

    units = pd.concat(tables, ignore_index=True)
    table_path = OUTPUT_DIR / f"good_cells_{SESSION}.csv"
    units.to_csv(table_path, index=False)
    print(f"\nSaved the full unit table to {table_path}")

    # Summary: how many units survive in each area, and why the rest failed.
    summary = units.groupby(["probe", "area"]).agg(
        units=("unit_id", "size"),
        good=("good", "sum"),
        failed_isi=("pass_isi", lambda s: int((~s).sum())),
        failed_spikes=("pass_spikes", lambda s: int((~s).sum())),
        failed_snr=("pass_snr", lambda s: int((~s).sum())),
    )
    good_units = units[units["good"]]
    summary["good_low_presence"] = good_units.groupby(["probe", "area"])["flag_low_presence"].sum()
    summary["good_possible_dup"] = good_units.groupby(["probe", "area"])["possible_duplicate_of"].count()
    summary = summary.fillna(0).astype(int)
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
