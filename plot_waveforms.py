"""Task 1: sparse waveform plots of good, rejected and flagged units.

Needs the table from good_cells.py. Run alone with `uv run plot_waveforms.py`, or both
tasks together with `uv run main.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.widgets as sw

from load_data import load_analyzer

SESSION = "2026-07-21"
AREAS = {"PFC": ["PFC"], "HPC": ["CA3", "DG"]}  # which areas to plot on each probe
UNITS_PER_FIGURE = 8


def note(unit):
    """Why the unit is in the figure, for the third line of its panel title."""
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
    """One panel per unit, on the sites near it. Faint lines: 50 spikes. Bold line: their average."""
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
    """One autocorrelogram per unit. One neuron leaves an empty gap around 0 ms (its refractory period)."""
    if rows.empty:
        return
    ccgs, bins = analyzer.get_extension("correlograms").get_data()
    unit_ids = list(analyzer.unit_ids)
    nrows = int(np.ceil(len(rows) / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 2.6 * nrows), squeeze=False)
    for ax, unit in zip(axes.flat, rows.itertuples()):
        k = unit_ids.index(unit.unit_id)  # position of this unit in the analyzer's arrays
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


def rows_in_order(unit_ids, probe_units):
    """The table rows for these units, in this order."""
    if not unit_ids:
        return probe_units.head(0)  # an empty table with the same columns
    return pd.concat([probe_units[probe_units.unit_id == unit_id] for unit_id in unit_ids])


def with_partners(flagged, partner_column, probe_units):
    """Each flagged unit followed by the unit named in partner_column, so pairs sit side by side."""
    order = []
    for _, unit in flagged.iterrows():
        order.append(unit.unit_id)
        order.append(unit[partner_column])
    return rows_in_order(order, probe_units)


def pick_rejected_examples(rejected):
    """Up to 2 random rejected units per fail reason, grouped by reason."""
    shuffled = rejected.sample(frac=1, random_state=0)  # random order, the same every run
    examples = shuffled.groupby("fail_reason").head(2)  # the first 2 of each reason
    return examples.sort_values("fail_reason")


def plot_area(analyzer, probe_units, area):
    """All figures for one brain area."""
    in_area = probe_units[probe_units.area == area]
    good = in_area[in_area.good]
    rejected = in_area[~in_area.good]
    prefix = f"outputs/waveforms_{SESSION}_{area}"

    # random_state fixes the sample so reruns show the same units
    good_sample = good.sample(min(UNITS_PER_FIGURE, len(good)), random_state=0)
    plot_units(
        analyzer, good_sample,
        f"{area}: random good units ({len(good)} of {len(in_area)} are good)", f"{prefix}_good.png",
    )
    plot_units(
        analyzer, pick_rejected_examples(rejected),
        f"{area}: rejected units, up to 2 per reason", f"{prefix}_rejected.png",
    )
    flagged = pd.concat([
        with_partners(in_area[in_area.same_neuron_as.notna()], "same_neuron_as", probe_units),
        with_partners(in_area[in_area.possible_duplicate_of.notna()], "possible_duplicate_of", probe_units),
        good[good.rescued_by_acg],
        good[good.low_presence].head(UNITS_PER_FIGURE),
    ])
    plot_units(
        analyzer, flagged,
        f"{area}: flagged units (duplicate pairs side by side, then rescued, then low presence)",
        f"{prefix}_flagged.png",
    )
    plot_acgs(
        analyzer, good[good.rescued_by_acg],
        f"{area}: autocorrelograms of units kept by the gap check", f"outputs/acg_{SESSION}_{area}_rescued.png",
    )


def main():
    units = pd.read_csv(f"outputs/good_cells_{SESSION}.csv")
    units["fail_reason"] = units.fail_reason.fillna("")  # good units have no fail reason
    for probe, areas in AREAS.items():
        analyzer = load_analyzer(SESSION, probe)  # slow, so load once per probe
        probe_units = units[units.probe == probe]
        for area in areas:
            plot_area(analyzer, probe_units, area)


if __name__ == "__main__":
    main()
