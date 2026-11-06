"""Firing rate and regularity (LvR, the local variation with refractoriness) of every unit, outside
the stretches replaced around artifact events."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import load_analyzer, load_artifact_windows
from sessions import get_session, session_from_command_line

LVR_R_MS = 5.0  # refractoriness constant R of the LvR (Shinomoto et al. 2009)


def inside_windows(spikes, starts, stops):
    """True for each spike (sample numbers in time order) inside a window, both ends included."""
    inside = np.zeros(len(spikes), dtype=bool)
    for start, stop in zip(starts, stops):
        first = np.searchsorted(spikes, start, side="left")  # index of the first spike at or after start
        last = np.searchsorted(spikes, stop, side="right")  # index of the first spike after stop
        inside[first:last] = True  # spikes first to last - 1 lie in [start, stop]
    return inside


def window_seconds(starts, stops, fs):
    """Total time inside the windows (s), counting time where windows overlap only once. The
    windows must be in time order, as load_artifact_windows returns them."""
    total_samples = 0
    current_start = None
    current_stop = None
    for start, stop in zip(starts, stops):
        if current_stop is not None and start <= current_stop + 1:  # overlaps or touches the current window
            current_stop = max(current_stop, stop)
        else:
            if current_stop is not None:
                total_samples += current_stop - current_start + 1
            current_start = start
            current_stop = stop
    if current_stop is not None:
        total_samples += current_stop - current_start + 1
    return total_samples / fs


def split_at_windows(kept, starts):
    """The kept spikes (in time order, none inside a window), split wherever a window starts
    between two consecutive spikes, so that no interval spans a window."""
    segments = []
    first = 0
    for start in starts:
        i = np.searchsorted(kept, start)  # index of the first kept spike after this window's start
        if i > first:
            segments.append(kept[first:i])
            first = i
    if first < len(kept):
        segments.append(kept[first:])
    return segments


def lvr(segments, fs):
    """LvR (Shinomoto et al. 2009) over every pair of consecutive intervals within a segment.
    Returns (lvr, pairs); lvr is NaN when there are no pairs.

    LvR = 3 / pairs * sum of (1 - 4 a b / (a + b)^2) (1 + 4 R / (a + b)) over consecutive intervals
    a and b (ms). It is near 1 for random firing with a refractory period of R (about 0.97 at a
    100 ms mean interval), lower for regular firing and higher for bursty firing.
    """
    total = 0.0
    pairs = 0
    for segment in segments:
        if len(segment) < 3:
            continue
        intervals = np.diff(segment) * 1000 / fs
        a = intervals[:-1]  # each interval...
        b = intervals[1:]  # ...and the one after it
        terms = (1 - 4 * a * b / (a + b) ** 2) * (1 + 4 * LVR_R_MS / (a + b))
        total += float(terms.sum())
        pairs += len(terms)
    if pairs == 0:
        return np.nan, 0
    return 3 * total / pairs, pairs


def firing_table(session, probe):
    """One row per unit of one probe: spikes inside the artifact windows, the time outside them,
    the firing rate over that time, and the LvR."""
    analyzer = load_analyzer(session, probe)
    fs = analyzer.sampling_frequency
    total_samples = analyzer.get_total_samples()
    starts, stops = load_artifact_windows(session, probe, fs, total_samples)
    time_outside_s = total_samples / fs - window_seconds(starts, stops, fs)
    rows = []
    for unit_id in analyzer.unit_ids:
        spikes = analyzer.sorting.get_unit_spike_train(unit_id)
        inside = inside_windows(spikes, starts, stops)
        kept = spikes[~inside]
        value, pairs = lvr(split_at_windows(kept, starts), fs)
        rows.append({
            "session": session,
            "probe": probe,
            "unit_id": unit_id,
            "spikes_in_windows": int(inside.sum()),
            "time_outside_windows_s": time_outside_s,
            "rate_hz": len(kept) / time_outside_s,
            "lvr": value,
            "lvr_pairs": pairs,
        })
    return pd.DataFrame(rows)


def plot_qc(firing, units, session):
    """LvR against firing rate for the main-analysis units, one panel per probe, coloured by area."""
    table = firing.merge(units[["probe", "unit_id", "area", "main_analysis"]], on=["probe", "unit_id"])
    table = table[table.main_analysis]
    probes = get_session(session)["probes"]
    fig, axes = plt.subplots(1, len(probes), figsize=(6 * len(probes), 4.5), squeeze=False)
    for ax, (probe, areas) in zip(axes.flat, probes.items()):
        for area in areas:
            in_area = table[(table.probe == probe) & (table.area == area)]
            ax.scatter(in_area.rate_hz, in_area.lvr, s=12, label=f"{area} ({len(in_area)})")
        ax.set_xscale("log")
        ax.axhline(1, color="gray", linestyle=":")
        ax.set_xlabel("firing rate outside the artifact windows (Hz)")
        ax.set_ylabel("LvR")
        ax.set_title(probe)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"outputs/firing_{session}.png", dpi=110)
    plt.close(fig)


def print_summary(firing):
    """Per probe: spikes left out inside the windows and the time outside them."""
    for probe, group in firing.groupby("probe", sort=False):
        print(
            f"{probe}: {group.spikes_in_windows.sum()} spikes inside the artifact windows; "
            f"{group.time_outside_windows_s.iloc[0]:.1f} s outside them"
        )


def main(session):
    Path("outputs").mkdir(exist_ok=True)
    tables = []
    for probe in get_session(session)["probes"]:
        tables.append(firing_table(session, probe))
    firing = pd.concat(tables, ignore_index=True)
    firing.to_csv(f"outputs/firing_{session}.csv", index=False)
    units = pd.read_csv(f"outputs/good_cells_{session}.csv")
    plot_qc(firing, units, session)
    print_summary(firing)


if __name__ == "__main__":
    main(session_from_command_line())
