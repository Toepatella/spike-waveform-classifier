"""Task 0: decide which units are good cells."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import area_of, load_analyzer, load_area_limits, load_metrics

SESSION = "2026-07-21"
PROBES = ["PFC", "HPC"]

# Good-cell rule: a clean average waveform from one neuron
MAX_CONTAMINATION = 0.1  # share of spikes from other neurons (Juan's rp_contamination, Llobet 2022)
MIN_SPIKES = 200  # enough spikes for a clean average
MIN_SNR = 3  # spike clearly above noise
# rp_contamination assumes a steady firing rate, so it over-rejects bursty (mostly
# narrow-spiking) cells. Rescue: keep a unit anyway if its own autocorrelogram shows a
# clean refractory gap, which is judged against its local firing rate. Rescued units are
# marked in rescued_by_acg, so later results can be checked with and without them.
MAX_ACG_GAP = 0.1  # autocorrelogram at |lag| < 1 ms, relative to |lag| > 10 ms (about <6% contamination)
MIN_ACG_BASELINE = 5  # spikes per 1 ms bin needed to trust the gap

# Duplicates: near-identical templates. One neuron split in two never fires twice within
# ~1 ms, so their cross-correlogram has a gap at zero lag. Keep one unit per neuron.
DUPLICATE_SIMILARITY = 0.95
SAME_NEURON_GAP = 0.3  # correlogram at |lag| < 1 ms, relative to |lag| > 10 ms
MIN_BASELINE = 2  # spikes per 1 ms bin needed to judge the gap

# Flag only (kept, check by eye): low-presence units fire unevenly across the
# session (silent for long stretches), not just slowly
LOW_PRESENCE = 0.5

Path("outputs").mkdir(exist_ok=True)

tables = []
for probe in PROBES:
    analyzer = load_analyzer(SESSION, probe)
    metrics = load_metrics(SESSION, probe).loc[analyzer.unit_ids]
    depth = analyzer.get_extension("unit_locations").get_data()[:, 1]
    limits = load_area_limits(SESSION, probe)

    # Polarity: sign of the biggest deflection on the unit's biggest channel. Positive spikes
    # are usually axonal or dendritic, and trough-based features don't fit them. Flag only.
    templates = analyzer.get_extension("templates").get_data(operator="average")  # units x time x channels
    biggest = np.abs(templates).max(axis=1).argmax(axis=1)
    trace = templates[np.arange(len(biggest)), :, biggest]
    polarity = np.where(np.abs(trace.min(axis=1)) >= np.abs(trace.max(axis=1)), "negative", "positive")

    similarity = analyzer.get_extension("template_similarity").get_data().copy()
    np.fill_diagonal(similarity, 0)
    ccgs, bins = analyzer.get_extension("correlograms").get_data()
    lag_ms = (bins[:-1] + bins[1:]) / 2

    # Each unit's refractory gap, from its autocorrelogram (its correlogram with itself)
    acgs = ccgs[np.arange(len(metrics)), np.arange(len(metrics))]
    acg_baseline = acgs[:, np.abs(lag_ms) > 10].mean(axis=1)
    acg_gap = acgs[:, np.abs(lag_ms) < 1].mean(axis=1) / np.maximum(acg_baseline, 1e-9)

    same_pairs = []
    possible_duplicate_of = [None] * len(metrics)  # too few spikes to tell
    for i, j in zip(*np.where(np.triu(similarity) > DUPLICATE_SIMILARITY)):
        near = ccgs[i, j, np.abs(lag_ms) < 1].mean()
        far = ccgs[i, j, np.abs(lag_ms) > 10].mean()
        if far < MIN_BASELINE:
            smaller, larger = sorted([i, j], key=lambda k: metrics.num_spikes.iloc[k])
            possible_duplicate_of[smaller] = metrics.index[larger]
        elif near < SAME_NEURON_GAP * far:
            same_pairs.append((i, j))

    table = pd.DataFrame({
        "probe": probe,
        "unit_id": metrics.index,
        "phy_cluster_id": analyzer.get_sorting_property("original_cluster_id"),
        "area": [area_of(d, limits) for d in depth],
        "depth_um": depth,
        "num_spikes": metrics.num_spikes.values,
        "rp_contamination": metrics.rp_contamination.values,
        "acg_gap": acg_gap,
        "isi_violations_ratio": metrics.isi_violations_ratio.values,
        "snr": metrics.snr.values,
        "presence_ratio": metrics.presence_ratio.values,
        "polarity": polarity,
        "possible_duplicate_of": possible_duplicate_of,
    })
    clean_acg = (acg_gap < MAX_ACG_GAP) & (acg_baseline >= MIN_ACG_BASELINE)
    table["rescued_by_acg"] = (table.rp_contamination > MAX_CONTAMINATION) & clean_acg
    # Written as "passes" so a missing value counts as a fail
    passes = pd.DataFrame({
        "contamination": (table.rp_contamination <= MAX_CONTAMINATION) | table.rescued_by_acg,
        "spikes": table.num_spikes >= MIN_SPIKES,
        "snr": table.snr >= MIN_SNR,
    })
    fails = ~passes

    # Units joined by same-neuron pairs are one neuron (it can be split in 3). Keep one piece:
    # one that passes, preferably without the gap rescue, then the one with most spikes.
    # Mark the others as copies of it and drop them.
    passes_all = passes.all(axis=1).values
    group = np.arange(len(table))
    for i, j in same_pairs:
        group[group == group[j]] = group[i]
    same_neuron_as = [None] * len(table)
    for g in {group[k] for pair in same_pairs for k in pair}:
        members = np.where(group == g)[0]
        keeper = max(
            members, key=lambda k: (passes_all[k], not table.rescued_by_acg[k], table.num_spikes[k])
        )
        for k in members[members != keeper]:
            same_neuron_as[k] = table.unit_id[keeper]
    table["same_neuron_as"] = same_neuron_as
    fails["duplicate"] = table.same_neuron_as.isin(table.unit_id[passes_all])
    table["fail_reason"] = fails.apply(lambda row: "+".join(row[row].index), axis=1)
    table["good"] = ~fails.any(axis=1)
    tables.append(table)

units = pd.concat(tables, ignore_index=True)
units["low_presence"] = units.presence_ratio < LOW_PRESENCE
units.to_csv(f"outputs/good_cells_{SESSION}.csv", index=False)

# Histograms with each cutoff drawn in red
panels = [
    ("rp_contamination", MAX_CONTAMINATION, np.linspace(0, 1, 51)),
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
    fig.savefig(f"outputs/metrics_{SESSION}_{probe}.png", dpi=110)
    plt.close(fig)

summary = units.assign(
    negative=units.good & (units.polarity == "negative"), rescued=units.good & units.rescued_by_acg
)
print(summary.groupby(["probe", "area"]).agg(
    units=("good", "size"),
    good=("good", "sum"),
    good_negative_spike=("negative", "sum"),
    good_by_acg_rescue=("rescued", "sum"),
))
