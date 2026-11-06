"""Checks on the pipeline's outputs and on its measurement code. Each check prints PASS, FAIL or
SKIP with what it found."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from features import bombcell_non_somatic, waveform_features
from firing import inside_windows, lvr, split_at_windows, window_seconds
from good_cells import contamination_from_pairs, one_piece_per_neuron
from load_data import chain_steps, load_analyzer, load_broadband_provenance
from sessions import get_session, session_from_command_line

# Copies of the unit table and metric histograms made with commit 0cb6002, before the session
# input, the new columns and Task 2 were added
REFERENCE_DIR = Path("outputs") / "reference_0cb6002"
FORMULA_TOLERANCE = 1e-12
BOUND_TOLERANCE = 1e-9

# The rules' numbers as the README states them, written out here rather than imported, so that a
# change to them in the code under test makes a check fail
NARROW_CUT_MS = 0.40
LIKELY_CONTAMINATED_CUT = 0.20
LVR_R_MS = 5.0
PEAK_FROM_MS = 0.1
PEAK_TO_MS = 1.5
PEAK_TOP_FRACTION = 0.9
UPSAMPLE = 10

# Synthetic waveforms: 120 samples at 30 kHz, sorted spike time at 2.0 ms, as in the stored templates
SYNTHETIC_FS = 30000.0
SYNTHETIC_SAMPLES = 120
SYNTHETIC_SORTED_MS = 2.0
TIME_TOLERANCE_MS = 0.002  # 2 us
DENSE_STEP_MS = 0.0001  # grid for the independent peak-time calculation


def print_result(name, passed, detail):
    """Print one check's result and return whether it passed."""
    if passed:
        word = "PASS"
    else:
        word = "FAIL"
    print(f"{word}  {name}: {detail}")
    return passed


def load_analyzers(session):
    """Each probe's analyzer, by probe name."""
    analyzers = {}
    for probe in get_session(session)["probes"]:
        analyzers[probe] = load_analyzer(session, probe)
    return analyzers


def check_regression(session):
    """Every column of the 0cb6002 reference unit table is unchanged, row by row, in the current
    table. Skipped for a session with no reference copy."""
    name = "unit table matches the 0cb6002 reference"
    reference_path = REFERENCE_DIR / f"good_cells_{session}.csv"
    if not reference_path.exists():
        print(f"SKIP  {name}: no reference copy for {session}")
        return True
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(Path("outputs") / f"good_cells_{session}.csv")
    if len(reference) != len(current):
        return print_result(name, False, f"{len(reference)} rows in the reference, {len(current)} now")
    changed = []
    for column in reference.columns:
        if column not in current.columns or not reference[column].equals(current[column]):
            changed.append(column)
    if changed:
        return print_result(name, False, "changed or missing columns: " + ", ".join(changed))
    added = []
    for column in current.columns:
        if column not in reference.columns:
            added.append(column)
    return print_result(name, True, f"{len(reference.columns)} columns identical; {len(added)} added")


def chance_above_count(mean, count):
    """P(X >= count) for a Poisson count X with this mean, minus 2.5%."""
    return stats.poisson.sf(count - 1, mean) - 0.025


def garwood_lower_mean(count):
    """The lowest Poisson mean consistent with a count at 95%: the mean at which a count this large
    or larger has a 2.5% chance, found by root-finding rather than the chi-square formula used in
    good_cells.py. 0 for a count of 0."""
    if count == 0:
        return 0.0
    return optimize.brentq(chance_above_count, 1e-12, count, args=(count,))


def check_lower_bound(session, analyzers):
    """The rp_contamination formula reproduces the stored rp_contamination from each unit's pair
    count, and rp_contamination_lower equals the formula applied to an independently found lower
    end of the pair count's 95% interval."""
    name = "contamination formula and lower bound"
    units = pd.read_csv(Path("outputs") / f"good_cells_{session}.csv")
    formula_difference = 0.0
    bound_difference = 0.0
    for unit in units.itertuples():
        analyzer = analyzers[unit.probe]
        total_samples = analyzer.get_total_samples()
        fs = analyzer.sampling_frequency
        estimate = contamination_from_pairs(unit.rp_violations, unit.num_spikes, total_samples, fs)
        formula_difference = max(formula_difference, abs(estimate - unit.rp_contamination))
        lowest_mean = garwood_lower_mean(int(unit.rp_violations))
        bound = contamination_from_pairs(lowest_mean, unit.num_spikes, total_samples, fs)
        bound_difference = max(bound_difference, abs(bound - unit.rp_contamination_lower))
    passed = formula_difference <= FORMULA_TOLERANCE and bound_difference <= BOUND_TOLERANCE
    detail = (
        f"largest difference from rp_contamination {formula_difference:.1e}; "
        f"from the recomputed lower bound {bound_difference:.1e}"
    )
    return print_result(name, passed, detail)


def gaussian(times_ms, centre_ms, width_ms):
    """A Gaussian bump of height 1."""
    return np.exp(-((times_ms - centre_ms) ** 2) / (2 * width_ms**2))


def sigmoid(times_ms, centre_ms, width_ms):
    """A smooth step from 0 to 1."""
    return 1 / (1 + np.exp(-(times_ms - centre_ms) / width_ms))


def tilted_plateau(times_ms):
    """A flat-topped bump from 2.35 to 2.85 ms, 6% higher at its start than at its end, so its
    maximum sits near the start rather than the middle of its top."""
    shape = sigmoid(times_ms, 2.35, 0.03) - sigmoid(times_ms, 2.85, 0.03)
    return 30 * shape * (1 + 0.06 * (2.85 - times_ms) / 0.5)


def double_bump(times_ms):
    """Two bumps 0.3 ms apart, the second 5% lower, with a deep dip between them."""
    return 30 * gaussian(times_ms, 2.45, 0.06) + 28.5 * gaussian(times_ms, 2.75, 0.06)


def top_centre(times_ms, values, trough_ms):
    """Independent version of the peak-time rule on a dense trace: (mean time of the samples from
    PEAK_FROM_MS to PEAK_TO_MS after the trough at or above PEAK_TOP_FRACTION of their maximum,
    time of that maximum)."""
    window = []
    for i in range(len(times_ms)):
        if trough_ms + PEAK_FROM_MS <= times_ms[i] <= trough_ms + PEAK_TO_MS:
            window.append(i)
    top = window[0]
    for i in window:
        if values[i] > values[top]:
            top = i
    top_times = []
    for i in window:
        if values[i] >= PEAK_TOP_FRACTION * values[top]:
            top_times.append(times_ms[i])
    return float(np.mean(top_times)), float(times_ms[top])


def peak_time_problems(times_ms, bump, label):
    """Compare the measured trough-to-peak of a Gaussian trough at 2.0 ms plus this bump with the
    independent calculation on the exact bump. Returns a list of problems.

    The expected value is computed on the feature code's own x10 time grid: on a peak with two
    humps, each grid sample entering or leaving the top moves the centre by several us.
    """
    spike = -100 * gaussian(times_ms, 2.0, 0.08) + bump(times_ms)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    upsampled_ms = np.arange(SYNTHETIC_SAMPLES * UPSAMPLE) * 1000 / (SYNTHETIC_FS * UPSAMPLE)
    centre_ms, maximum_ms = top_centre(upsampled_ms, bump(upsampled_ms), 2.0)
    problems = []
    # written as "not <=" so that a missing (NaN) value fails too
    if not abs(features["trough_to_peak_ms"] - (centre_ms - 2.0)) <= TIME_TOLERANCE_MS:
        problems.append(f"{label}: trough-to-peak {features['trough_to_peak_ms']:.4f} ms, expected {centre_ms - 2.0:.4f}")
    dense_ms = np.arange(1.5, 3.6, DENSE_STEP_MS)
    dense_centre_ms, dense_maximum_ms = top_centre(dense_ms, bump(dense_ms), 2.0)
    if abs(dense_maximum_ms - dense_centre_ms) < 0.03:
        problems.append(f"{label}: the test shape can't tell the rule from the time of the maximum")
    return problems


def check_features_synthetic():
    """The feature code finds known landmarks on synthetic 30 kHz waveforms."""
    name = "features on synthetic waveforms"
    times_ms = np.arange(SYNTHETIC_SAMPLES) * 1000 / SYNTHETIC_FS
    problems = []

    # A Gaussian trough at 2.0 ms (width 0.08 ms) and a Gaussian peak 0.6 ms later (width 0.15 ms).
    # rising_phase_ms isn't checked: the peak's tail moves its 10% crossing by about 2 us.
    spike = -100 * gaussian(times_ms, 2.0, 0.08) + 30 * gaussian(times_ms, 2.6, 0.15)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    expected = {
        "trough_to_peak_ms": 0.6,
        "half_width_ms": 2 * np.sqrt(2 * np.log(2)) * 0.08,  # full width at half depth of a Gaussian
        "falling_phase_ms": np.sqrt(2 * np.log(10)) * 0.08,  # a Gaussian is at 10% of its depth this far from its centre
        "peak_decay_ms": np.sqrt(2 * np.log(2)) * 0.15,  # a Gaussian is at half its height this far from its centre
        "repolarization_ms": 0.15,  # the falling branch's steepest point, one width after the centre
    }
    for feature, value in expected.items():
        if not abs(features[feature] - value) <= TIME_TOLERANCE_MS:
            problems.append(f"{feature} {features[feature]:.4f} ms, expected {value:.4f}")
    if not abs(features["peak_trough_ratio"] - 0.3) <= 0.01:
        problems.append(f"peak_trough_ratio {features['peak_trough_ratio']:.3f}, expected 0.300")
    if bombcell_non_somatic(features):
        problems.append("a spike with no pre-trough peak is flagged by the Bombcell rule")

    # Flat-topped and double-humped peaks: the peak time is the centre of the samples at 90% or more
    problems = problems + peak_time_problems(times_ms, tilted_plateau, "flat peak")
    problems = problems + peak_time_problems(times_ms, double_bump, "double peak")

    # A trough that recovers slowly from below the baseline, with no positive peak after it
    spike = -100 * gaussian(times_ms, 2.0, 0.08) - 20 * gaussian(times_ms, 2.6, 0.3)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    if "no_positive_peak" not in features["missing"] or not np.isnan(features["trough_to_peak_ms"]):
        problems.append(f"no peak: missing '{features['missing']}', trough-to-peak {features['trough_to_peak_ms']}")

    # A trace still rising at the end of the peak search
    spike = -100 * gaussian(times_ms, 2.0, 0.08) + 30 * sigmoid(times_ms, 3.3, 0.15)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    if "peak_at_window_end" not in features["missing"]:
        problems.append(f"rising tail: missing '{features['missing']}'")

    # A peak at 0.5 ms whose top also includes a plateau reaching the end of the search, beyond
    # which the top may go on
    spike = -100 * gaussian(times_ms, 2.0, 0.08) + 30 * gaussian(times_ms, 2.5, 0.1) + 29.5 * sigmoid(times_ms, 3.4, 0.02)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    if "peak_at_window_end" not in features["missing"]:
        problems.append(f"top at the window end: missing '{features['missing']}'")

    # A fast rebound that is already falling when the search starts, 0.1 ms after the trough
    spike = -100 * gaussian(times_ms, 2.0, 0.03) + 40 * gaussian(times_ms, 2.06, 0.03)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    if "peak_at_window_start" not in features["missing"]:
        problems.append(f"top at the window start: missing '{features['missing']}'")

    # A narrow, tall pre-trough peak before a narrow trough: Bombcell's non-somatic shape
    spike = 40 * gaussian(times_ms, 1.8, 0.04) - 100 * gaussian(times_ms, 2.0, 0.05) + 5 * gaussian(times_ms, 2.5, 0.1)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    if not bombcell_non_somatic(features):
        problems.append("the Bombcell shape is not flagged")
    if not abs(features["pre_peak_ratio"] - 0.40) <= 0.01:
        problems.append(f"pre_peak_ratio {features['pre_peak_ratio']:.3f}, expected 0.400")
    expected_width = 2 * np.sqrt(2 * np.log(2)) * 0.04  # full width at half height of the pre-trough Gaussian
    if not abs(features["pre_peak_width_ms"] - expected_width) <= 0.003:
        problems.append(f"pre_peak_width_ms {features['pre_peak_width_ms']:.4f}, expected {expected_width:.4f}")

    # The same pre-trough peak with a tail still rising at the end of the peak search: not flagged,
    # since the post-trough peak is unknown
    spike = 40 * gaussian(times_ms, 1.8, 0.04) - 100 * gaussian(times_ms, 2.0, 0.05) + 30 * sigmoid(times_ms, 3.3, 0.1)
    features, landmarks = waveform_features(spike, SYNTHETIC_FS, SYNTHETIC_SORTED_MS)
    if bombcell_non_somatic(features):
        problems.append("a pre-trough peak with an unknown post-trough peak is flagged")

    if problems:
        return print_result(name, False, "; ".join(problems))
    return print_result(name, True, "nine synthetic waveforms measured as expected")


def check_split_neuron_keeper():
    """good_cells.one_piece_per_neuron keeps one piece of a split neuron in a version even when the
    piece with the most spikes fails that version."""
    name = "one piece per split neuron"
    # Pieces 0, 1 and 2 of one neuron. Piece 0 has the most spikes but fails this version (for the
    # SNR-only check: it passes the main rule only through the amplitude clause); 1 and 2 pass it.
    passes = np.array([False, True, True])
    passes_strict = np.array([False, False, False])
    kept = one_piece_per_neuron([[0, 1, 2]], passes, passes_strict, [1000, 900, 800])
    kept_list = []
    for value in kept:
        kept_list.append(bool(value))
    return print_result(name, kept_list == [False, True, False], f"kept {kept_list}, expected [False, True, False]")


def check_features_table(session, analyzers):
    """The feature table has one row per unit, in the unit table's order. Every downward unit has a
    trough-to-peak or a code saying why not, every upward unit has the code positive_spike, each
    unit was measured on its channel with the largest |value|, and the decay and repolarization
    times, measured after the peak, are positive."""
    name = "feature table"
    units = pd.read_csv(Path("outputs") / f"good_cells_{session}.csv")
    features = pd.read_csv(Path("outputs") / f"features_{session}.csv")
    if len(units) != len(features):
        return print_result(name, False, f"{len(units)} units but {len(features)} feature rows")
    same_probes = (units.probe.values == features.probe.values).all()
    same_units = (units.unit_id.values == features.unit_id.values).all()
    if not (same_probes and same_units):
        return print_result(name, False, "rows are not in the unit table's order")
    templates = {}
    unit_positions = {}
    for probe, analyzer in analyzers.items():
        templates[probe] = analyzer.get_extension("templates").get_data(operator="average")
        unit_positions[probe] = list(analyzer.unit_ids)
    unexplained = 0
    upward_wrong = 0
    wrong_channel = 0
    not_positive = 0
    for unit, row in zip(units.itertuples(), features.itertuples()):
        missing = row.missing
        if not isinstance(missing, str):  # an empty code reads back from the CSV as NaN
            missing = ""
        if unit.polarity == "positive":
            if missing != "positive_spike":
                upward_wrong += 1
        elif np.isnan(row.trough_to_peak_ms) and missing == "":
            unexplained += 1
        template = templates[unit.probe][unit_positions[unit.probe].index(unit.unit_id)]
        if row.channel != int(np.abs(template).max(axis=0).argmax()):
            wrong_channel += 1
        if row.peak_decay_ms <= 0 or row.repolarization_ms <= 0:  # NaN compares as False
            not_positive += 1
    passed = unexplained == 0 and upward_wrong == 0 and wrong_channel == 0 and not_positive == 0
    detail = (
        f"{len(features)} rows; downward units with no value and no code: {unexplained}; "
        f"upward units without positive_spike: {upward_wrong}; wrong channel: {wrong_channel}; "
        f"decay or repolarization not positive: {not_positive}"
    )
    return print_result(name, passed, detail)


def lvr_term_by_term(spikes, fs):
    """LvR of one unbroken spike train, computed one interval pair at a time, separately from
    firing.lvr."""
    intervals = []
    for i in range(1, len(spikes)):
        intervals.append((spikes[i] - spikes[i - 1]) * 1000 / fs)
    total = 0.0
    for i in range(len(intervals) - 1):
        a = intervals[i]
        b = intervals[i + 1]
        total += (1 - 4 * a * b / (a + b) ** 2) * (1 + 4 * LVR_R_MS / (a + b))
    return 3 * total / (len(intervals) - 1)


def check_lvr_synthetic():
    """firing.py's LvR and artifact-window handling on spike trains with known answers."""
    name = "LvR and artifact windows on synthetic spike trains"
    fs = SYNTHETIC_FS
    problems = []

    # Random firing with a 5 ms refractory period and a 100 ms mean interval: matches the
    # term-by-term formula, and LvR is about 0.97, a little below 1
    generator = np.random.default_rng(0)
    intervals_ms = LVR_R_MS + generator.exponential(95.0, 2000)
    times_ms = np.cumsum(intervals_ms)  # spike times
    spikes = np.round(times_ms * fs / 1000).astype(np.int64)  # in samples
    value, pairs = lvr([spikes], fs)
    reference = lvr_term_by_term(spikes, fs)
    if abs(value - reference) > 1e-9:
        problems.append(f"random train: LvR {value:.6f}, term by term {reference:.6f}")
    if not 0.9 <= value <= 1.05:
        problems.append(f"random train: LvR {value:.3f}, expected about 0.97")

    # Perfectly regular firing: 1000 spikes 100 ms apart (3000 samples at 30 kHz)
    regular = np.arange(1, 1001) * 3000
    value, pairs = lvr([regular], fs)
    if abs(value) > 1e-12:
        problems.append(f"regular train: LvR {value}, expected 0")

    # One window over the middle spike: that spike is dropped and no interval spans the window
    spikes = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    starts = np.array([250], dtype=np.int64)
    stops = np.array([350], dtype=np.int64)
    inside = inside_windows(spikes, starts, stops)
    segments = split_at_windows(spikes[~inside], starts)
    value, pairs = lvr(segments, fs)
    segment_lists = []
    for segment in segments:
        segment_lists.append(segment.tolist())
    if inside.tolist() != [False, False, True, False, False]:
        problems.append(f"window: inside {inside.tolist()}")
    if segment_lists != [[100, 200], [400, 500]]:
        problems.append(f"window: segments {segment_lists}")
    if pairs != 0 or not np.isnan(value):
        problems.append(f"window: {pairs} interval pairs used, expected none")
    if abs(window_seconds(starts, stops, fs) - 101 / fs) > 1e-12:
        problems.append("window: length should be 101 samples")

    # Two overlapping windows count their shared time once
    if abs(window_seconds(np.array([100, 150]), np.array([200, 260]), fs) - 161 / fs) > 1e-12:
        problems.append("overlapping windows: length should be 161 samples")

    if problems:
        return print_result(name, False, "; ".join(problems))
    return print_result(name, True, "random, regular and windowed trains handled as expected")


def covered_samples(session, probe, total_samples, fs):
    """True for each sample of the recording inside a stretch replaced around an artifact event,
    rebuilt from the recording's provenance without using firing.py or load_artifact_windows."""
    covered = np.zeros(total_samples, dtype=bool)
    for name, settings in chain_steps(load_broadband_provenance(session, probe)):
        if name == "RemoveArtifactsRecording":
            before = int(settings["ms_before"] * fs / 1000)
            after = int(settings["ms_after"] * fs / 1000)
            for event in settings["list_triggers"][0]:
                first = max(event - before, 0)
                last = min(event + after, total_samples - 1)
                covered[first:last + 1] = True
    return covered


def lvr_from_labels(kept, labels, fs):
    """LvR and its number of interval pairs, using only the pairs of consecutive intervals whose
    three spikes share a label (no replaced stretch between them). Written separately from
    firing.lvr and firing.split_at_windows."""
    intervals = np.diff(kept) * 1000 / fs
    a = intervals[:-1]
    b = intervals[1:]
    same_stretch = labels[:-2] == labels[2:]  # the first and third spike of each pair of intervals
    terms = (1 - 4 * a * b / (a + b) ** 2) * (1 + 4 * LVR_R_MS / (a + b))
    pairs = int(same_stretch.sum())
    if pairs == 0:
        return np.nan, 0
    return 3 * float(terms[same_stretch].sum()) / pairs, pairs


def check_firing(session, analyzers):
    """The firing table, recounted from the provenance and the spike trains: its rows follow the
    unit table, each probe's time outside the artifact windows is right, and each unit's spikes
    inside the windows, its rate and its LvR are right."""
    name = "firing table"
    units = pd.read_csv(Path("outputs") / f"good_cells_{session}.csv")
    firing = pd.read_csv(Path("outputs") / f"firing_{session}.csv")
    if len(units) != len(firing):
        return print_result(name, False, f"{len(units)} units but {len(firing)} firing rows")
    same_probes = (units.probe.values == firing.probe.values).all()
    same_units = (units.unit_id.values == firing.unit_id.values).all()
    if not (same_probes and same_units):
        return print_result(name, False, "rows are not in the unit table's order")
    problems = []
    for probe, analyzer in analyzers.items():
        fs = analyzer.sampling_frequency
        total_samples = analyzer.get_total_samples()
        covered = covered_samples(session, probe, total_samples, fs)
        covered_so_far = np.cumsum(covered)  # covered samples up to each sample
        time_outside_s = (total_samples - covered.sum()) / fs
        probe_firing = firing[firing.probe == probe]
        probe_units = units[units.probe == probe]
        if (probe_firing.time_outside_windows_s - time_outside_s).abs().max() > 1e-9:
            problems.append(f"{probe}: time outside the windows")
        for row, unit in zip(probe_firing.itertuples(), probe_units.itertuples()):
            spikes = analyzer.sorting.get_unit_spike_train(row.unit_id)
            spike_covered = covered[spikes]  # whether each spike's sample is covered
            kept = spikes[~spike_covered]
            # two kept spikes have no replaced stretch between them exactly when the count of
            # covered samples is the same at both
            value, pairs = lvr_from_labels(kept, covered_so_far[kept], fs)
            if len(spikes) != unit.num_spikes:
                problems.append(f"{probe} unit {row.unit_id}: spike count")
            if int(spike_covered.sum()) != row.spikes_in_windows:
                problems.append(f"{probe} unit {row.unit_id}: spikes inside the windows")
            if abs(row.rate_hz * time_outside_s - len(kept)) > 1e-6:
                problems.append(f"{probe} unit {row.unit_id}: rate")
            if pairs != row.lvr_pairs:
                problems.append(f"{probe} unit {row.unit_id}: LvR interval pairs")
            elif pairs > 0 and abs(value - row.lvr) > 1e-9:
                problems.append(f"{probe} unit {row.unit_id}: LvR")
    if problems:
        return print_result(name, False, f"{len(problems)} problems, first: {problems[0]}")
    return print_result(name, True, f"{len(firing)} rows; windows, spikes inside them, rates and LvR recounted")


def check_report(session):
    """Every row of the narrow-share report, recomputed from the unit and feature tables (whose
    rows match, as the feature-table check confirms)."""
    name = "narrow-share report"
    units = pd.read_csv(Path("outputs") / f"good_cells_{session}.csv")
    features = pd.read_csv(Path("outputs") / f"features_{session}.csv")
    shares = pd.read_csv(Path("outputs") / f"narrow_shares_{session}.csv")
    main = units.main_analysis
    masks = {
        "main": main,
        "strict check": units.main_analysis_strict,
        "SNR-only check": units.main_analysis_snr_only,
        "main without likely contaminated": main & (units.rp_contamination_lower <= LIKELY_CONTAMINATED_CUT),
        "main without near white matter": main & (units.near_wm_limit == False),
        "main without Bombcell non-somatic": main & (features.bombcell_non_somatic == False),
    }
    rows_expected = 0
    mismatches = []
    for version, members in masks.items():
        for probe, areas in get_session(session)["probes"].items():
            for area in areas:
                rows_expected += 1
                in_group = members & (units.probe == probe) & (units.area == area)
                ttp = features.trough_to_peak_ms[in_group]
                measured = int(ttp.notna().sum())
                narrow = int((ttp < NARROW_CUT_MS).sum())
                found = shares[(shares.version == version) & (shares.probe == probe) & (shares.area == area)]
                if len(found) != 1:
                    mismatches.append(f"{version} {probe} {area}: {len(found)} rows")
                    continue
                row = found.iloc[0]
                count = int(in_group.sum())
                same_counts = row.units == count and row.no_ttp == count - measured and row.narrow == narrow
                if measured > 0:
                    same_share = abs(row.narrow_share - round(narrow / measured, 3)) < 1e-9
                else:
                    same_share = np.isnan(row.narrow_share)
                if not (same_counts and same_share):
                    mismatches.append(f"{version} {probe} {area}")
    if len(shares) != rows_expected:
        mismatches.append(f"{len(shares)} rows, expected {rows_expected}")
    if mismatches:
        return print_result(name, False, "; ".join(mismatches))
    return print_result(name, True, f"all {rows_expected} rows recomputed")


def check_no_hardcoded_session():
    """No script holds a quoted date such as a session name; sessions are listed only in
    sessions.py."""
    name = "no session named outside sessions.py"
    quoted_date = re.compile(r"[\"']\d{4}-\d{2}-\d{2}[\"']")
    offenders = []
    for path in sorted(Path(".").glob("*.py")):
        if path.name == "sessions.py":
            continue
        if quoted_date.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    if offenders:
        return print_result(name, False, ", ".join(offenders))
    return print_result(name, True, "none found")


def main(session):
    analyzers = load_analyzers(session)
    results = [
        check_regression(session),
        check_lower_bound(session, analyzers),
        check_split_neuron_keeper(),
        check_features_synthetic(),
        check_features_table(session, analyzers),
        check_lvr_synthetic(),
        check_firing(session, analyzers),
        check_report(session),
        check_no_hardcoded_session(),
    ]
    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} checks passed or were skipped")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main(session_from_command_line())
