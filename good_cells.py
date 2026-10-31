"""Task 0: decide which units are good cells.

Run alone with `uv run good_cells.py`, or together with Task 1 with `uv run main.py`.
"""

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
MAX_ACG_GAP = 0.1  # spike pairs under 1 ms apart, relative to 10-25 ms apart (about <6% contamination)
MIN_ACG_BASELINE = 5  # spikes per 1 ms bin needed to trust the gap

# Duplicates: near-identical templates. One neuron split in two never fires twice within
# ~1 ms, so their cross-correlogram has a gap at zero lag. Keep one unit per neuron.
DUPLICATE_SIMILARITY = 0.95
SAME_NEURON_GAP = 0.3  # the same gap measure as MAX_ACG_GAP, for a pair of units
MIN_BASELINE = 2  # spikes per 1 ms bin needed to judge the gap

# Flag only (kept, check by eye): low-presence units fire unevenly across the
# session (silent for long stretches), not just slowly
LOW_PRESENCE = 0.5


def get_polarity(templates):
    """'negative' or 'positive' for each unit: the sign of the biggest deflection on its biggest channel.

    Positive spikes are usually axonal or dendritic, and trough-based features don't fit them.
    """
    polarity = []
    for template in templates:  # one unit: time x channels
        biggest_channel = np.abs(template).max(axis=0).argmax()
        trace = template[:, biggest_channel]
        if abs(trace.min()) >= abs(trace.max()):
            polarity.append("negative")
        else:
            polarity.append("positive")
    return polarity


def refractory_gap(correlogram, lag_ms):
    """How empty a correlogram is around 0 ms.

    Returns (gap, baseline). The baseline is the average count 10-25 ms from zero. The gap is
    the average count within 1 ms of zero divided by the baseline: 0 means a clean refractory
    gap, about 1 means no gap. A small baseline means too few spikes to trust the gap.
    """
    baseline = correlogram[np.abs(lag_ms) > 10].mean()
    near_zero = correlogram[np.abs(lag_ms) < 1].mean()
    if baseline == 0:
        return np.nan, baseline
    return near_zero / baseline, baseline


def find_duplicate_pairs(similarity, ccgs, lag_ms, num_spikes, unit_ids):
    """Check every pair of units with near-identical templates.

    Returns:
      same_neuron_pairs: (i, j) positions of pairs with a clear gap in their cross-correlogram
      possible_duplicate_of: for pairs with too few spikes to tell, the smaller unit's entry
        holds the larger unit's ID (None everywhere else)
    """
    same_neuron_pairs = []
    possible_duplicate_of = [None] * len(unit_ids)
    for i in range(len(unit_ids)):
        for j in range(i + 1, len(unit_ids)):
            if similarity[i, j] <= DUPLICATE_SIMILARITY:
                continue
            gap, baseline = refractory_gap(ccgs[i, j], lag_ms)
            if baseline < MIN_BASELINE:
                if num_spikes[i] <= num_spikes[j]:
                    possible_duplicate_of[i] = unit_ids[j]
                else:
                    possible_duplicate_of[j] = unit_ids[i]
            elif gap < SAME_NEURON_GAP:
                same_neuron_pairs.append((i, j))
    return same_neuron_pairs, possible_duplicate_of


def group_split_neurons(same_neuron_pairs, num_units):
    """Join same-neuron pairs into one group per neuron: pairs (A, B) and (B, C) make [A, B, C].

    Returns a list of groups, each a list of unit positions. Units not in any pair are left out.
    """
    group_of = list(range(num_units))  # every unit starts in its own group
    for i, j in same_neuron_pairs:
        old_group = group_of[j]
        for k in range(num_units):
            if group_of[k] == old_group:
                group_of[k] = group_of[i]

    groups = {}
    for k in range(num_units):
        if group_of[k] not in groups:
            groups[group_of[k]] = []
        groups[group_of[k]].append(k)
    return [members for members in groups.values() if len(members) > 1]


def choose_keeper(members, passes, rescued, num_spikes):
    """Which piece of a split neuron to keep: one that passes the rule, preferably without the
    gap rescue, then the one with the most spikes."""
    candidates = [k for k in members if passes[k]]
    if not candidates:
        candidates = members
    without_rescue = [k for k in candidates if not rescued[k]]
    if without_rescue:
        candidates = without_rescue
    return max(candidates, key=lambda k: num_spikes[k])


def build_unit_table(probe):
    """One row per unit of one probe: its measures, flags and the good-cell decision."""
    analyzer = load_analyzer(SESSION, probe)
    metrics = load_metrics(SESSION, probe).loc[analyzer.unit_ids]  # same unit order as the analyzer
    unit_ids = list(metrics.index)
    num_spikes = metrics.num_spikes.values
    depth = analyzer.get_extension("unit_locations").get_data()[:, 1]  # y: position along the probe (um)
    limits = load_area_limits(SESSION, probe)
    templates = analyzer.get_extension("templates").get_data(operator="average")  # units x time x channels
    similarity = analyzer.get_extension("template_similarity").get_data()  # units x units
    ccgs, bins = analyzer.get_extension("correlograms").get_data()  # units x units x 1 ms bins
    lag_ms = (bins[:-1] + bins[1:]) / 2  # centre of each bin

    # Each unit's own refractory gap, from its autocorrelogram (its correlogram with itself)
    acg_gap = []
    clean_acg = []
    for k in range(len(unit_ids)):
        gap, baseline = refractory_gap(ccgs[k, k], lag_ms)
        acg_gap.append(gap)
        clean_acg.append(gap < MAX_ACG_GAP and baseline >= MIN_ACG_BASELINE)

    same_neuron_pairs, possible_duplicate_of = find_duplicate_pairs(
        similarity, ccgs, lag_ms, num_spikes, unit_ids
    )

    table = pd.DataFrame({
        "probe": probe,
        "unit_id": unit_ids,
        "phy_cluster_id": analyzer.get_sorting_property("original_cluster_id"),
        "area": [area_of(d, limits) for d in depth],
        "depth_um": depth,
        "num_spikes": num_spikes,
        "rp_contamination": metrics.rp_contamination.values,
        "acg_gap": acg_gap,
        "isi_violations_ratio": metrics.isi_violations_ratio.values,
        "snr": metrics.snr.values,
        "presence_ratio": metrics.presence_ratio.values,
        "polarity": get_polarity(templates),
        "possible_duplicate_of": possible_duplicate_of,
    })
    # Rescued: Juan's score says contaminated, but the unit's own autocorrelogram has a clean gap
    table["rescued_by_acg"] = (table.rp_contamination > MAX_CONTAMINATION) & np.array(clean_acg)

    # The rule, one column per criterion. Written as "passes" so a missing value counts as a fail.
    passes = pd.DataFrame({
        "contamination": (table.rp_contamination <= MAX_CONTAMINATION) | table.rescued_by_acg,
        "spikes": table.num_spikes >= MIN_SPIKES,
        "snr": table.snr >= MIN_SNR,
    })
    passes_all = passes.all(axis=1).values

    # Split neurons: keep one piece of each and mark the others as copies of it
    same_neuron_as = [None] * len(table)
    for members in group_split_neurons(same_neuron_pairs, len(table)):
        keeper = choose_keeper(members, passes_all, table.rescued_by_acg.values, num_spikes)
        for k in members:
            if k != keeper:
                same_neuron_as[k] = unit_ids[keeper]
    table["same_neuron_as"] = same_neuron_as

    fails = ~passes
    # A copy is dropped only if the piece it copies is kept
    fails["duplicate"] = table.same_neuron_as.isin(table.unit_id[passes_all])

    # e.g. "contamination+snr"; empty for good units
    fail_reason = []
    for k in range(len(table)):
        failed = [name for name in fails.columns if fails.loc[k, name]]
        fail_reason.append("+".join(failed))
    table["fail_reason"] = fail_reason
    table["good"] = ~fails.any(axis=1)
    return table


def plot_histograms(units):
    """Histograms of each measure with its cutoff in red, one figure per probe."""
    panels = [
        # (column, cutoff, bin edges)
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


def print_summary(units):
    """Units, good units, good negative-spike units and rescued good units per area."""
    for (probe, area), group in units.groupby(["probe", "area"]):
        good = group[group.good]
        negative = good[good.polarity == "negative"]
        rescued = good[good.rescued_by_acg]
        print(
            f"{probe} {area:13}  units {len(group):3}  good {len(good):3}  "
            f"good negative-spike {len(negative):3}  good by gap rescue {len(rescued):2}"
        )


def main():
    Path("outputs").mkdir(exist_ok=True)
    tables = []
    for probe in PROBES:
        tables.append(build_unit_table(probe))
    units = pd.concat(tables, ignore_index=True)
    units["low_presence"] = units.presence_ratio < LOW_PRESENCE
    units.to_csv(f"outputs/good_cells_{SESSION}.csv", index=False)
    plot_histograms(units)
    print_summary(units)


if __name__ == "__main__":
    main()
