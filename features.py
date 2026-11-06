"""Task 2: waveform features of each unit's stored average template, measured after x10
interpolation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from good_cells import largest_channel
from load_data import load_analyzer
from sessions import get_session, session_from_command_line

# Interpolation: x10, assuming the template holds no frequencies above half its sampling rate
# (scipy's resample_poly: a 201-tap filter spanning 10 original samples on each side, with zeros
# beyond the template's ends). The filter is tapered by a Kaiser window with beta 10: on a
# synthetic spike, SciPy's default beta 5 leaves ripples of up to 0.07% of the trough depth next
# to fast changes, beta 10 about 0.0004%.
UPSAMPLE = 10
UPSAMPLE_WINDOW = ("kaiser", 10.0)

# The sorted spike time is the time the sorter gave the spike, 2.0 ms into each stored template.
TROUGH_SEARCH_MS = 0.3  # trough: the minimum within this of the sorted spike time
# Baseline: the median of the template's first 0.5 ms (2.0-1.5 ms before the sorted spike time),
# a median 0.1% of the trough depth from zero in the first session's main units. The review's
# provisional window, 0.8-0.5 ms before the trough, sits on the positive bump before the trough
# instead: a median 9% of the trough depth, and above 10% in about half of the main units.
BASELINE_MS = 0.5
PEAK_FROM_MS = 0.1  # post-trough peak: the maximum from this long after the trough...
PEAK_TO_MS = 1.5  # ...to this long after it
PEAK_TOP_FRACTION = 0.9  # peak time: the centre of the samples at or above this share of the maximum
PHASE_LEVEL = -0.1  # falling and rising phases: crossings of 10% of the trough depth
FALL_SLOPE_MS = 0.5  # fall_slope: the steepest descent in this long before the trough
PRE_PEAK_MS = 1.0  # pre-trough peak: the largest value in this long before the trough
VECTOR_FROM_MS = -0.5  # waveform vector: from this long before the trough...
VECTOR_TO_MS = 1.5  # ...to this long after it...
VECTOR_STEP_MS = 0.01  # ...on a 100 kHz grid
NARROW_TTP_MS = 0.40  # narrow spikes: trough-to-peak below this (provisional)

# How far each template must reach before and after the sorted spike time for the windows above
# (the trough can be up to TROUGH_SEARCH_MS early or late)
MIN_BEFORE_MS = BASELINE_MS + PRE_PEAK_MS + TROUGH_SEARCH_MS
MIN_AFTER_MS = TROUGH_SEARCH_MS + PEAK_TO_MS

# Bombcell's non-somatic rule, converted from samples to ms in the 2026-11-05 review of the
# preprocessing: a narrow, tall pre-trough peak before a narrow trough
BOMBCELL_MAX_PRE_PEAK_WIDTH_MS = 0.15
BOMBCELL_MAX_TROUGH_WIDTH_MS = 0.2
BOMBCELL_MIN_PRE_PEAK_RATIO = 0.2
BOMBCELL_MIN_PRE_OVER_POST = 3.0

FEATURES = [
    "trough_to_peak_ms", "half_width_ms", "peak_trough_ratio", "peak_decay_ms", "repolarization_ms",
    "falling_phase_ms", "rising_phase_ms", "fall_slope", "rise_slope", "recovery_slope",
    "pre_peak_ratio", "pre_peak_width_ms",
]
QC_UNITS = 12  # units drawn in each area's QC figure


def upsample(trace, fs):
    """x10 interpolation of one trace. Returns (times_ms, values); values[10 * k] falls at the time
    of original sample k."""
    values = signal.resample_poly(trace, UPSAMPLE, 1, window=UPSAMPLE_WINDOW)  # up by 10, down by 1
    times_ms = np.arange(len(values)) * 1000 / (fs * UPSAMPLE)
    return times_ms, values


def index_at(times_ms, time_ms):
    """Index of the sample nearest to time_ms, kept inside the trace."""
    step_ms = times_ms[1] - times_ms[0]
    index = int(round((time_ms - times_ms[0]) / step_ms))
    return min(max(index, 0), len(times_ms) - 1)


def find_trough(times_ms, values, sorted_ms):
    """Index of the minimum within TROUGH_SEARCH_MS of the sorted spike time."""
    start = index_at(times_ms, sorted_ms - TROUGH_SEARCH_MS)
    stop = index_at(times_ms, sorted_ms + TROUGH_SEARCH_MS)
    return start + int(np.argmin(values[start:stop + 1]))  # + 1: a slice leaves out its end


def baseline_level(trace, fs):
    """The median of the template's first BASELINE_MS, on the original samples (uV)."""
    count = int(round(BASELINE_MS * fs / 1000))
    return float(np.median(trace[:count]))


def normalize(values, trough_i, baseline):
    """The trace with its baseline at 0 and its trough at -1, or None if the trough isn't below
    the baseline."""
    depth = baseline - values[trough_i]
    if depth <= 0:
        return None
    return (values - baseline) / depth


def crossing_time(times_ms, y, level, start, step):
    """Time of the first crossing of level, walking from index start in direction step (+1 or -1),
    by linear interpolation between samples. None if the trace never crosses it."""
    i = start
    while 0 <= i + step < len(y):
        a = y[i]
        b = y[i + step]
        if (a - level) * (b - level) <= 0 and a != b:  # a and b lie on either side of level, or on it
            fraction = (level - a) / (b - a)
            return times_ms[i] + fraction * (times_ms[i + step] - times_ms[i])
        i += step
    return None


def find_peak(times_ms, y, trough_i):
    """The post-trough peak: (index of its maximum, peak time in ms, index of its last top sample,
    missing code).

    The maximum is sought from PEAK_FROM_MS to PEAK_TO_MS after the trough. The top is the samples
    in that window at or above PEAK_TOP_FRACTION of the maximum. The peak time is their centre
    (mean time), which shifts less with noise than the time of the maximum when the top is flat;
    the peak's decay is measured after the last top sample. The peak can't be measured if the
    trace never rises above the baseline in the window (no_positive_peak), or if the top reaches
    either end of the window, so that part of it may lie outside (peak_at_window_start,
    peak_at_window_end).
    """
    trough_ms = times_ms[trough_i]
    start = index_at(times_ms, trough_ms + PEAK_FROM_MS)
    stop = index_at(times_ms, trough_ms + PEAK_TO_MS)
    peak_i = start + int(np.argmax(y[start:stop + 1]))
    if y[peak_i] <= 0:
        return peak_i, np.nan, peak_i, "no_positive_peak"
    level = PEAK_TOP_FRACTION * y[peak_i]
    top_times = []
    first_top_i = None
    last_top_i = peak_i
    for i in range(start, stop + 1):
        if y[i] >= level:
            top_times.append(times_ms[i])
            if first_top_i is None:
                first_top_i = i
            last_top_i = i
    if last_top_i == stop:
        return peak_i, np.nan, last_top_i, "peak_at_window_end"
    if first_top_i == start:
        return peak_i, np.nan, last_top_i, "peak_at_window_start"
    return peak_i, float(np.mean(top_times)), last_top_i, ""


def first_return_to_zero(y, start):
    """Index where the trace first falls back to 0 after index start, or the last index."""
    i = start
    while i + 1 < len(y):
        i += 1
        if y[i] <= 0:
            return i
    return i


def steepest_slope(times_ms, y, start, stop, direction):
    """Steepest slope between indices start and stop, in trough depths per ms: the most negative
    for direction "down", the most positive for "up". Returns (slope, time of the middle of that
    step in ms), or (nan, nan) if the range holds no step."""
    if stop <= start:
        return np.nan, np.nan
    step_ms = times_ms[1] - times_ms[0]
    slopes = np.diff(y[start:stop + 1]) / step_ms
    if direction == "down":
        j = int(np.argmin(slopes))
    else:
        j = int(np.argmax(slopes))
    middle_ms = (times_ms[start + j] + times_ms[start + j + 1]) / 2
    return float(slopes[j]), float(middle_ms)


def empty_features(code):
    """Features of a unit that can't be measured: every value NaN, with the reason as its code."""
    features = {}
    for name in FEATURES:
        features[name] = np.nan
    features["bombcell_non_somatic"] = False
    features["missing"] = code
    return features


def measure_half_width(times_ms, y, trough_i, features, landmarks):
    """Width at half the trough depth. Returns the missing codes."""
    left = crossing_time(times_ms, y, -0.5, trough_i, -1)
    right = crossing_time(times_ms, y, -0.5, trough_i, +1)
    if left is None or right is None:
        return ["no_half_width"]
    features["half_width_ms"] = right - left
    landmarks["half_left_ms"] = left
    landmarks["half_right_ms"] = right
    return []


def measure_phases(times_ms, y, trough_i, features):
    """The falling phase (from 10% of the trough depth down to the trough), the rising phase (from
    the trough back to 10%), and the steepest descent before the trough. Returns the missing codes."""
    codes = []
    trough_ms = times_ms[trough_i]
    fall_start = crossing_time(times_ms, y, PHASE_LEVEL, trough_i, -1)
    if fall_start is None:
        codes.append("no_fall_start")
    else:
        features["falling_phase_ms"] = trough_ms - fall_start
    rise_end = crossing_time(times_ms, y, PHASE_LEVEL, trough_i, +1)
    if rise_end is None:
        codes.append("no_rise_end")
    else:
        features["rising_phase_ms"] = rise_end - trough_ms
    start = index_at(times_ms, trough_ms - FALL_SLOPE_MS)
    slope, slope_ms = steepest_slope(times_ms, y, start, trough_i, "down")
    features["fall_slope"] = slope
    return codes


def measure_pre_trough_peak(times_ms, y, trough_i, features):
    """The largest value in the PRE_PEAK_MS before the trough, kept even when it is 0 or negative,
    and, if it is positive, its width at half its height. Returns the missing codes."""
    start = index_at(times_ms, times_ms[trough_i] - PRE_PEAK_MS)
    if trough_i <= start:
        return ["no_pre_peak"]
    pre_i = start + int(np.argmax(y[start:trough_i]))
    features["pre_peak_ratio"] = y[pre_i]
    if y[pre_i] <= 0:
        return ["no_pre_peak"]
    left = crossing_time(times_ms, y, y[pre_i] / 2, pre_i, -1)
    right = crossing_time(times_ms, y, y[pre_i] / 2, pre_i, +1)
    if left is None or right is None:
        return ["no_pre_peak"]
    features["pre_peak_width_ms"] = right - left
    return []


def measure_post_trough_peak(times_ms, y, trough_i, features, landmarks):
    """Trough-to-peak, the peak's height, the steepest rise to it, its decay to half its height,
    and the steepest descent on its falling branch. Returns the missing codes."""
    peak_i, peak_ms, last_top_i, code = find_peak(times_ms, y, trough_i)
    if code:
        return [code]
    codes = []
    landmarks["peak_ms"] = peak_ms
    features["trough_to_peak_ms"] = peak_ms - times_ms[trough_i]
    features["peak_trough_ratio"] = y[peak_i]  # the trough is -1, so this is peak height over trough depth
    slope, slope_ms = steepest_slope(times_ms, y, trough_i, peak_i, "up")
    features["rise_slope"] = slope
    half_ms = crossing_time(times_ms, y, y[peak_i] / 2, last_top_i, +1)
    if half_ms is None:
        codes.append("decay_outside_window")
    else:
        features["peak_decay_ms"] = half_ms - peak_ms
    end = first_return_to_zero(y, last_top_i)
    slope, slope_ms = steepest_slope(times_ms, y, last_top_i, end, "down")
    if np.isnan(slope) or slope >= 0:
        codes.append("no_falling_branch")
    else:
        features["recovery_slope"] = slope
        features["repolarization_ms"] = slope_ms - peak_ms
    return codes


def waveform_features(trace, fs, sorted_ms):
    """Features of one downward waveform (1-D, uV), measured on its normalized x10 trace.

    Returns (features, landmarks). features holds FEATURES and missing: the codes of the landmarks
    that weren't found, joined by ";". A missing landmark leaves its features empty (NaN); nothing
    is filled in. landmarks holds the normalized trace and the times the features were measured
    at, for the QC figure.
    """
    features = empty_features("")
    trace = np.asarray(trace, dtype=float)
    times_ms, values = upsample(trace, fs)
    trough_i = find_trough(times_ms, values, sorted_ms)
    y = normalize(values, trough_i, baseline_level(trace, fs))
    landmarks = {
        "times_ms": times_ms, "y": y, "trough_ms": times_ms[trough_i],
        "peak_ms": np.nan, "half_left_ms": np.nan, "half_right_ms": np.nan,
    }
    if y is None:
        features["missing"] = "no_trough"
        return features, landmarks
    codes = measure_half_width(times_ms, y, trough_i, features, landmarks)
    codes = codes + measure_phases(times_ms, y, trough_i, features)
    codes = codes + measure_pre_trough_peak(times_ms, y, trough_i, features)
    codes = codes + measure_post_trough_peak(times_ms, y, trough_i, features, landmarks)
    features["missing"] = ";".join(codes)
    return features, landmarks


def bombcell_non_somatic(features):
    """Bombcell's non-somatic rule: a pre-trough peak narrower than 0.15 ms, at least 0.2 of the
    trough's depth and at least 3 times the post-trough peak (when that peak is positive), before a
    trough with a half-width under 0.2 ms. False when a needed landmark is missing, or when the
    post-trough peak's top reaches an end of its search window, since its height is then unknown."""
    pre_ratio = features["pre_peak_ratio"]
    pre_width = features["pre_peak_width_ms"]
    trough_width = features["half_width_ms"]
    if np.isnan(pre_ratio) or np.isnan(pre_width) or np.isnan(trough_width):
        return False
    if pre_width >= BOMBCELL_MAX_PRE_PEAK_WIDTH_MS:
        return False
    if trough_width >= BOMBCELL_MAX_TROUGH_WIDTH_MS:
        return False
    if pre_ratio < BOMBCELL_MIN_PRE_PEAK_RATIO:
        return False
    if "peak_at_window" in features["missing"]:  # peak_at_window_start or peak_at_window_end
        return False
    post = features["peak_trough_ratio"]
    if np.isnan(post):  # no post-trough peak above the baseline
        post = 0.0
    if post > 0 and pre_ratio < BOMBCELL_MIN_PRE_OVER_POST * post:
        return False
    return True


def vector_times():
    """Times of the waveform vector's points, relative to the trough (ms)."""
    count = int(round((VECTOR_TO_MS - VECTOR_FROM_MS) / VECTOR_STEP_MS)) + 1
    return np.linspace(VECTOR_FROM_MS, VECTOR_TO_MS, count)


def waveform_vector(landmarks):
    """The normalized trace around the trough on the vector grid, or NaN values if the trough
    wasn't found."""
    grid_ms = vector_times()
    if landmarks["y"] is None:
        return np.full(len(grid_ms), np.nan)
    return np.interp(landmarks["trough_ms"] + grid_ms, landmarks["times_ms"], landmarks["y"])


def measure_probe(session, probe, units):
    """Features of one probe's units, in the unit table's order.

    Returns (table, vectors, landmarks): one table row and one waveform vector per unit, and the
    landmarks of each downward unit by unit ID. Upward units get NaN features and the code
    positive_spike. Stops the run if the stored templates are too short for the measurements.
    """
    analyzer = load_analyzer(session, probe)
    templates_extension = analyzer.get_extension("templates")
    templates = templates_extension.get_data(operator="average")  # units x time x channels
    fs = analyzer.sampling_frequency
    sorted_ms = templates_extension.nbefore * 1000 / fs  # the sorted spike time within each template
    after_ms = (templates.shape[1] - templates_extension.nbefore) * 1000 / fs
    if sorted_ms < MIN_BEFORE_MS or after_ms < MIN_AFTER_MS:
        raise SystemExit(
            f"{session} {probe}: templates span {sorted_ms:.2f} ms before and {after_ms:.2f} ms after the "
            f"sorted spike time; the features need {MIN_BEFORE_MS:.1f} ms before and {MIN_AFTER_MS:.1f} ms after"
        )
    unit_ids = list(analyzer.unit_ids)
    rows = []
    vectors = []
    landmarks_by_unit = {}
    for unit in units.itertuples():
        k = unit_ids.index(unit.unit_id)  # position of this unit in the analyzer's arrays
        channel = largest_channel(templates[k])  # its position in the analyzer's channel list
        row = {
            "session": session, "probe": probe, "unit_id": unit.unit_id,
            "phy_cluster_id": unit.phy_cluster_id, "area": unit.area, "channel": channel,
        }
        if unit.polarity == "positive":
            features = empty_features("positive_spike")
            vector = np.full(len(vector_times()), np.nan)
        else:
            features, landmarks = waveform_features(templates[k][:, channel], fs, sorted_ms)
            features["bombcell_non_somatic"] = bombcell_non_somatic(features)
            vector = waveform_vector(landmarks)
            landmarks_by_unit[unit.unit_id] = landmarks
        row.update(features)
        rows.append(row)
        vectors.append(vector)
    return pd.DataFrame(rows), vectors, landmarks_by_unit


def plot_qc(features, landmarks_by_unit, main_ids, session, probe, area):
    """One figure per area for checking the measurement code: trough-to-peak of the main-analysis
    units with the narrow cut, and 12 of their normalized traces with the trough, the peak time and
    the half-width crossings marked."""
    in_area = features[(features.area == area) & features.unit_id.isin(main_ids)]
    measured = in_area.trough_to_peak_ms.notna().sum()
    fig = plt.figure(figsize=(16, 7))
    left, right = fig.subfigures(1, 2, width_ratios=[1, 2])
    ax = left.subplots()
    ax.hist(in_area.trough_to_peak_ms.dropna(), bins=np.arange(0, 1.25, 0.02), color="gray")
    ax.axvline(NARROW_TTP_MS, color="red", linestyle="--")
    ax.set_xlabel("trough-to-peak (ms)")
    ax.set_title(f"{probe} {area}: {measured} main-analysis units with a trough-to-peak")
    axes = right.subplots(3, 4, sharex=True, sharey=True)
    sample = in_area.sample(min(QC_UNITS, len(in_area)), random_state=0)  # the same units every run
    for ax, unit in zip(axes.flat, sample.itertuples()):
        marks = landmarks_by_unit[unit.unit_id]
        if marks["y"] is None:  # no trough below baseline: nothing to draw
            continue
        times = marks["times_ms"] - marks["trough_ms"]
        ax.plot(times, marks["y"], color="black", linewidth=0.8)
        ax.axvline(0, color="blue", linewidth=1.0)
        if not np.isnan(marks["peak_ms"]):
            ax.axvline(marks["peak_ms"] - marks["trough_ms"], color="red", linewidth=1.0)
        if not np.isnan(marks["half_left_ms"]):
            half = [marks["half_left_ms"] - marks["trough_ms"], marks["half_right_ms"] - marks["trough_ms"]]
            ax.plot(half, [-0.5, -0.5], color="green", linewidth=1.5)
        ax.set_xlim(-1, 2)
        ax.set_title(
            f"unit {unit.unit_id} (Phy {unit.phy_cluster_id})  trough-to-peak {unit.trough_to_peak_ms:.3f}",
            fontsize=7,
        )
    for ax in axes.flat[len(sample):]:
        ax.set_visible(False)
    right.suptitle("blue: trough, red: peak time, green: half-width (time in ms from the trough)", fontsize=9)
    fig.savefig(f"outputs/features_{session}_{probe}_{area}.png", dpi=110)
    plt.close(fig)


def print_summary(features, main_ids):
    """For one probe, per area: main-analysis units, how many have a trough-to-peak, how many are
    narrow, the Bombcell count and the missing-landmark codes."""
    main = features[features.unit_id.isin(main_ids)]
    for (probe, area), group in main.groupby(["probe", "area"]):
        measured = group.trough_to_peak_ms.notna()
        narrow = (group.trough_to_peak_ms < NARROW_TTP_MS).sum()
        codes = {}
        for text in group.missing:
            for code in text.split(";"):
                if code:
                    codes[code] = codes.get(code, 0) + 1
        print(
            f"{probe} {area:13}  main {len(group):3}  with trough-to-peak {measured.sum():3}  "
            f"narrow {narrow:3}  Bombcell {group.bombcell_non_somatic.sum():2}  missing {codes}"
        )


def main(session):
    Path("outputs").mkdir(exist_ok=True)
    units = pd.read_csv(f"outputs/good_cells_{session}.csv")
    tables = []
    vectors = []
    for probe, areas in get_session(session)["probes"].items():
        probe_units = units[units.probe == probe]
        table, probe_vectors, landmarks_by_unit = measure_probe(session, probe, probe_units)
        tables.append(table)
        vectors = vectors + probe_vectors
        main_ids = probe_units.unit_id[probe_units.main_analysis]  # unit IDs are unique within a probe
        for area in areas:
            plot_qc(table, landmarks_by_unit, main_ids, session, probe, area)
        print_summary(table, main_ids)
    features = pd.concat(tables, ignore_index=True)
    features.to_csv(f"outputs/features_{session}.csv", index=False)
    np.savez(
        f"outputs/waveform_vectors_{session}.npz",
        vectors=np.array(vectors),
        times_ms=vector_times(),
        probe=features.probe.to_numpy(dtype=str),  # plain text, so np.load needs no pickle
        unit_id=features.unit_id.to_numpy(),
    )


if __name__ == "__main__":
    main(session_from_command_line())
