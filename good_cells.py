"""Task 0: decide which units are good cells."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import area_of, load_analyzer, load_area_limits, load_metrics

SESSION = "2026-07-21"
PROBES = ["PFC", "HPC"]

# Good-cell rule
MAX_ISI = 0.5     # contamination from other neurons
MIN_SPIKES = 200  # enough spikes for a clean average
MIN_SNR = 3       # spike clearly above noise

# Flags only (kept, check by eye)
LOW_PRESENCE = 0.5
DUPLICATE_SIMILARITY = 0.95

Path("outputs").mkdir(exist_ok=True)

tables = []
for probe in PROBES:
    analyzer = load_analyzer(SESSION, probe)
    metrics = load_metrics(SESSION, probe).loc[analyzer.unit_ids]
    depth = analyzer.get_extension("unit_locations").get_data()[:, 1]
    limits = load_area_limits(SESSION, probe)

    # Possible duplicates: near-identical templates. Flag the one with fewer spikes.
    similarity = analyzer.get_extension("template_similarity").get_data().copy()
    np.fill_diagonal(similarity, 0)
    duplicate_of = [None] * len(metrics)
    for i, j in zip(*np.where(np.triu(similarity) > DUPLICATE_SIMILARITY)):
        smaller, larger = sorted([i, j], key=lambda k: metrics.num_spikes.iloc[k])
        duplicate_of[smaller] = metrics.index[larger]

    tables.append(pd.DataFrame({
        "probe": probe,
        "unit_id": metrics.index,
        "area": [area_of(d, limits) for d in depth],
        "depth_um": depth,
        "num_spikes": metrics.num_spikes.values,
        "isi_violations_ratio": metrics.isi_violations_ratio.values,
        "snr": metrics.snr.values,
        "presence_ratio": metrics.presence_ratio.values,
        "possible_duplicate_of": duplicate_of,
    }))

units = pd.concat(tables, ignore_index=True)
units["good"] = (
    (units.isi_violations_ratio < MAX_ISI)
    & (units.num_spikes >= MIN_SPIKES)
    & (units.snr >= MIN_SNR)
)
units["low_presence"] = units.presence_ratio < LOW_PRESENCE
units.to_csv(f"outputs/good_cells_{SESSION}.csv", index=False)

# Histograms with each cutoff drawn in red
panels = [
    ("isi_violations_ratio", MAX_ISI, np.linspace(0, 3, 50)),
    ("num_spikes", MIN_SPIKES, np.logspace(1, 5, 50)),
    ("snr", MIN_SNR, np.linspace(0, 15, 50)),
    ("presence_ratio", LOW_PRESENCE, np.linspace(0, 1, 50)),
]
for probe in PROBES:
    probe_units = units[units.probe == probe]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, (column, cutoff, bins) in zip(axes, panels):
        # clip so values past the axis pile up at the edge instead of vanishing
        ax.hist(probe_units[column].clip(bins[0], bins[-1]), bins=bins, color="gray")
        ax.axvline(cutoff, color="red", linestyle="--")
        ax.set_title(column)
    axes[1].set_xscale("log")
    fig.suptitle(f"{probe}: {probe_units.good.sum()} of {len(probe_units)} good")
    fig.tight_layout()
    fig.savefig(f"outputs/metrics_{probe}.png", dpi=110)
    plt.close(fig)

print(units.groupby(["probe", "area"]).good.agg(units="size", good="sum"))
