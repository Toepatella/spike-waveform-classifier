"""Task 1: sparse waveform plots of good, rejected and flagged units. Run good_cells.py first."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.widgets as sw

from load_data import load_analyzer

SESSION = "2026-07-21"
AREAS = {"PFC": ["PFC"], "HPC": ["CA3", "DG"]}
UNITS_PER_FIGURE = 8

units = pd.read_csv(f"outputs/good_cells_{SESSION}.csv").fillna({"fail_reason": ""})


def note(unit):
    # Why the unit is in the figure
    notes = []
    if unit.fail_reason:
        notes.append(f"fails: {unit.fail_reason}")
    if unit.rescued_by_acg:
        notes.append(f"clean autocorrelogram (gap {unit.acg_gap:.2f})")
    if pd.notna(unit.same_neuron_as):
        notes.append(f"same neuron as {unit.same_neuron_as:.0f}")
    if pd.notna(unit.possible_duplicate_of):
        notes.append(f"possible duplicate of {unit.possible_duplicate_of:.0f}")
    if unit.low_presence:
        notes.append(f"presence {unit.presence_ratio:.2f}")
    return ", ".join(notes)


def plot_units(analyzer, rows, title, path):
    # One panel per unit on the sites near it. Faint lines: 50 spikes. Bold line: their average.
    if rows.empty:
        return
    nrows = int(np.ceil(len(rows) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 4.5 * nrows), squeeze=False)
    sw.plot_unit_waveforms(
        analyzer, unit_ids=list(rows.unit_id), axes=list(axes.flat[: len(rows)]), max_spikes_per_unit=50
    )
    for ax, unit in zip(axes.flat, rows.itertuples()):
        ax.set_title(
            f"unit {unit.unit_id} (Phy {unit.phy_cluster_id}), {unit.polarity}\n"
            f"SNR {unit.snr:.1f}  contam {unit.rp_contamination:.2f}  spikes {unit.num_spikes}\n{note(unit)}",
            fontsize=7,
        )
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_acgs(analyzer, rows, title, path):
    # Autocorrelograms: one neuron leaves an empty gap around 0 ms (its refractory period)
    if rows.empty:
        return
    ccgs, bins = analyzer.get_extension("correlograms").get_data()
    index = {unit_id: k for k, unit_id in enumerate(analyzer.unit_ids)}
    nrows = int(np.ceil(len(rows) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 2.6 * nrows), squeeze=False)
    for ax, unit in zip(axes.flat, rows.itertuples()):
        k = index[unit.unit_id]
        ax.bar(bins[:-1], ccgs[k, k], width=np.diff(bins), align="edge", color="gray")
        ax.set_title(
            f"unit {unit.unit_id} (Phy {unit.phy_cluster_id})\n"
            f"Juan's contam {unit.rp_contamination:.2f}  gap {unit.acg_gap:.2f}",
            fontsize=7,
        )
        ax.set_xlabel("lag (ms)", fontsize=7)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def with_partners(flagged, column, probe_units):
    # Each flagged unit followed by the unit it may duplicate, so pairs sit side by side
    rows = []
    for unit in flagged.itertuples():
        rows += [flagged.loc[[unit.Index]], probe_units[probe_units.unit_id == getattr(unit, column)]]
    return pd.concat(rows) if rows else flagged


for probe, areas in AREAS.items():
    analyzer = load_analyzer(SESSION, probe)
    probe_units = units[units.probe == probe]
    for area in areas:
        in_area = probe_units[probe_units.area == area]
        good, rejected = in_area[in_area.good], in_area[~in_area.good]
        figures = {
            # random_state fixes the samples so reruns show the same units
            "good": (
                good.sample(min(UNITS_PER_FIGURE, len(good)), random_state=0),
                f"{area}: random good units ({len(good)} of {len(in_area)} are good)",
            ),
            "rejected": (
                rejected.sample(frac=1, random_state=0).groupby("fail_reason").head(2).sort_values("fail_reason"),
                f"{area}: rejected units, up to 2 per reason",
            ),
            "flagged": (
                pd.concat([
                    with_partners(in_area[in_area.same_neuron_as.notna()], "same_neuron_as", probe_units),
                    with_partners(in_area[in_area.possible_duplicate_of.notna()], "possible_duplicate_of", probe_units),
                    good[good.rescued_by_acg],
                    good[good.low_presence].head(UNITS_PER_FIGURE),
                ]),
                f"{area}: flagged units (duplicate pairs side by side, then rescued, then low presence)",
            ),
        }
        for name, (rows, title) in figures.items():
            plot_units(analyzer, rows, title, f"outputs/waveforms_{SESSION}_{area}_{name}.png")
        plot_acgs(
            analyzer,
            good[good.rescued_by_acg],
            f"{area}: autocorrelograms of units kept by the gap check",
            f"outputs/acg_{SESSION}_{area}_rescued.png",
        )
