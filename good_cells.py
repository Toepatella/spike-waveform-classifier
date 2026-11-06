"""Task 0: decide which units are good cells."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from load_data import area_of, load_analyzer, load_area_limits, load_metrics
from sessions import get_session, session_from_command_line

# Good-cell rule: a clean average waveform from one neuron. All cuts were recommended with the
# sorted data (2026-11-02; the contamination cut was confirmed at review on 2026-11-04).
# rp_contamination compares a unit's spike pairs 0.2-1 ms apart (closer pairs were removed as
# duplicates) with the number expected if spikes from other neurons arrived at a steady rate,
# independent of the unit. When they rise and fall with the unit's own firing, the score reads
# high. The expected number grows with the square of the firing rate, so below about 0.7 Hz a
# single pair already gives 0.5 or more: slow units pass only with none.
MAX_CONTAMINATION = 0.5  # share of spikes from other neurons (rp_contamination, Llobet 2022); below this
STRICT_CONTAMINATION = 0.1  # strict check (good_strict): at most this
MIN_SPIKES = 500  # need more than this; the average waveforms use up to 500 random spikes
MIN_SNR = 3  # spike clearly above the channel's noise...
MIN_AMPLITUDE_UV = 30  # ...or bigger than this (uV), so big units on noisy channels still count

# Lower bound of rp_contamination, not part of the rule: the lower end of its exact 95% interval.
# For a unit below MAX_CONTAMINATION it can exceed 0.20 only with many spikes (from about 7,600 in
# a 52-minute session).
REFRACTORY_MS = 1.0  # rp_violations: spike pairs at most this far apart
CENSOR_MS = 0.2  # spikes closer than this within a unit were removed as duplicates
INTERVAL_LEVEL = 0.95  # exact (Garwood) Poisson interval on rp_violations

# Near-limit sensitivity, not part of the rule: the area limits are estimates, so units just above
# the top of the white matter may be white-matter units.
NEAR_LIMIT_UM = 100
WHITE_MATTER = "White Matter"

# Duplicates: near-identical templates. One neuron split in two never fires twice within
# ~1 ms, so their cross-correlogram has a gap at zero lag. Keep one unit per neuron.
DUPLICATE_SIMILARITY = 0.95
SAME_NEURON_GAP = 0.3  # refractory_gap of the pair's cross-correlogram
MIN_BASELINE = 2  # spikes per 1 ms bin needed to judge the gap

# Flag only, used by no version: low-presence units fire unevenly across the session (silent for
# long stretches), not just slowly
LOW_PRESENCE = 0.5

# Manual review (rule fixed on 2026-11-03, before the review and before any Task 2 results):
# verdicts on units checked by eye are typed into the session's review file. "exclude" removes a
# unit from every version; "keep" and "unsure" leave it in. The review can only remove units,
# never add them.
REVIEW_VERDICTS = ["keep", "exclude", "unsure"]


def review_file(session):
    """The session's manual review file."""
    return Path("curation") / f"manual_review_{session}.csv"


def largest_channel(template):
    """The channel with the largest |value| of one unit's template (time x channels): the channel
    the polarity rule and the waveform features use."""
    biggest_per_channel = np.abs(template).max(axis=0)  # each channel's largest |value| over time
    return int(biggest_per_channel.argmax())


def get_polarity(templates):
    """'negative' or 'positive' for each unit: the sign of the biggest deflection on its biggest channel.

    Positive spikes are usually axonal or dendritic, and trough-based features don't fit them.
    """
    polarity = []
    for template in templates:  # one unit: time x channels
        trace = template[:, largest_channel(template)]
        if abs(trace.min()) >= abs(trace.max()):
            polarity.append("negative")
        else:
            polarity.append("positive")
    return polarity


def contamination_from_pairs(num_pairs, num_spikes, total_samples, fs):
    """Share of spikes from other neurons, c, implied by a number of refractory pairs (the count
    itself, or an end of its confidence interval), by the rp_contamination formula
    pairs = c (2 - c) N^2 (t_r - t_c) / (T - 2 N t_c), with N spikes, T samples in the recording,
    and t_r and t_c the refractory and censored periods in samples."""
    t_r = round(REFRACTORY_MS * fs / 1000)
    t_c = round(CENSOR_MS * fs / 1000)
    # the pairs expected if every spike came from other neurons (c = 1)
    pairs_if_all_stray = num_spikes**2 * (t_r - t_c) / (total_samples - 2 * num_spikes * t_c)
    ratio = num_pairs / pairs_if_all_stray
    if ratio >= 1:
        return 1.0
    return 1 - np.sqrt(1 - ratio)  # c (2 - c) = ratio, solved for c


def contamination_lower_bound(num_pairs, num_spikes, total_samples, fs):
    """Lower end of the 95% confidence interval of rp_contamination: the lowest Poisson mean
    consistent with the pair count, put through the same formula. 0 for a unit with no pairs."""
    if num_pairs == 0:
        return 0.0
    # Garwood (1936): the lowest Poisson mean consistent with num_pairs at 95% is half the 2.5%
    # point of a chi-square distribution with 2 x num_pairs degrees of freedom
    lowest_mean_pairs = stats.chi2.ppf((1 - INTERVAL_LEVEL) / 2, 2 * num_pairs) / 2
    return contamination_from_pairs(lowest_mean_pairs, num_spikes, total_samples, fs)


def near_white_matter(depth_um, limits):
    """True for a unit outside the white matter within NEAR_LIMIT_UM above its top. Always False on
    a probe with no white-matter limit."""
    if WHITE_MATTER not in limits:
        return False
    top = limits[WHITE_MATTER]
    return top < depth_um <= top + NEAR_LIMIT_UM


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


def choose_keeper(members, passes, passes_strict, num_spikes):
    """Which piece of a split neuron to keep: one that passes the rule, preferably one that also
    passes the strict check, then the one with the most spikes."""
    candidates = [k for k in members if passes[k]]
    if not candidates:
        candidates = members
    strict = [k for k in candidates if passes_strict[k]]
    if strict:
        candidates = strict
    return max(candidates, key=lambda k: num_spikes[k])


def one_piece_per_neuron(groups, passes, passes_strict, num_spikes):
    """For one version of the rule: True for each unit that passes it, except that each split neuron
    keeps only one piece, chosen by choose_keeper among the pieces that pass this version."""
    kept = list(passes)
    for members in groups:
        keeper = choose_keeper(members, passes, passes_strict, num_spikes)
        for k in members:
            if k != keeper:
                kept[k] = False
    return kept


def load_review_verdicts(session, probe, table):
    """Each unit's status in the manual review file, in table order.

    "" means not listed for review, "pending" means listed but not reviewed yet, and otherwise
    it's the verdict typed into the file. A listed unit whose Phy ID and unit ID don't match the
    table, or an unknown verdict, stops the run, so a mixed-up ID can't exclude the wrong unit.
    """
    path = review_file(session)
    status = [""] * len(table)
    if not path.exists():
        return status
    review = pd.read_csv(path, dtype={"verdict": str, "note": str})
    for row in review.itertuples():
        if row.probe != probe:
            continue
        positions = list(table.index[table.phy_cluster_id == row.phy_cluster_id])
        if len(positions) != 1 or table.unit_id[positions[0]] != row.unit_id:
            raise ValueError(
                f"{path}: no unit on the {probe} probe has Phy ID {row.phy_cluster_id} "
                f"and unit ID {row.unit_id}"
            )
        if pd.isna(row.verdict) or row.verdict.strip() == "":
            verdict = "pending"
        else:
            verdict = row.verdict.strip().lower()
        if verdict != "pending" and verdict not in REVIEW_VERDICTS:
            raise ValueError(f"{path}: unknown verdict '{row.verdict}' for Phy {row.phy_cluster_id}")
        status[positions[0]] = verdict
    return status


def build_unit_table(session, probe):
    """One row per unit of one probe: its measures, flags and the good-cell decision."""
    analyzer = load_analyzer(session, probe)
    metrics = load_metrics(session, probe).loc[analyzer.unit_ids]  # same unit order as the analyzer
    unit_ids = list(metrics.index)
    num_spikes = metrics.num_spikes.values
    rp_violations = metrics.rp_violations.values
    total_samples = analyzer.get_total_samples()
    fs = analyzer.sampling_frequency
    depth = analyzer.get_extension("unit_locations").get_data()[:, 1]  # y: position along the probe (um)
    limits = load_area_limits(session, probe)
    templates = analyzer.get_extension("templates").get_data(operator="average")  # units x time x channels
    similarity = analyzer.get_extension("template_similarity").get_data()  # units x units
    ccgs, bins = analyzer.get_extension("correlograms").get_data()  # units x units x 1 ms bins
    lag_ms = (bins[:-1] + bins[1:]) / 2  # centre of each bin

    # Each unit's own refractory gap, from its autocorrelogram (its correlogram with itself).
    # Descriptive only, not part of the rule.
    acg_gap = []
    for k in range(len(unit_ids)):
        gap, baseline = refractory_gap(ccgs[k, k], lag_ms)
        acg_gap.append(gap)

    # The lower end of each unit's contamination interval, from its refractory pairs
    rp_contamination_lower = []
    for k in range(len(unit_ids)):
        bound = contamination_lower_bound(rp_violations[k], num_spikes[k], total_samples, fs)
        rp_contamination_lower.append(bound)

    same_neuron_pairs, possible_duplicate_of = find_duplicate_pairs(
        similarity, ccgs, lag_ms, num_spikes, unit_ids
    )

    table = pd.DataFrame({
        "session": session,
        "probe": probe,
        "unit_id": unit_ids,
        "phy_cluster_id": analyzer.get_sorting_property("original_cluster_id"),
        "area": [area_of(d, limits) for d in depth],
        "depth_um": depth,
        "near_wm_limit": [near_white_matter(d, limits) for d in depth],
        "num_spikes": num_spikes,
        "firing_rate": metrics.firing_rate.values,  # spikes per second over the session
        "rp_contamination": metrics.rp_contamination.values,
        "rp_contamination_lower": rp_contamination_lower,  # lower end of its exact 95% interval; not in the rule
        "rp_violations": rp_violations,  # spike pairs 0.2-1 ms apart
        "acg_gap": acg_gap,
        "isi_violations_ratio": metrics.isi_violations_ratio.values,
        "snr": metrics.snr.values,
        "amplitude_median": metrics.amplitude_median.values,  # biggest deflection of the average spike (uV)
        # separation from neighbouring units by waveform alone; falls with spike size, not used in the rule
        "nn_hit_rate": metrics.nn_hit_rate.values,
        "presence_ratio": metrics.presence_ratio.values,
        "polarity": get_polarity(templates),
        "possible_duplicate_of": possible_duplicate_of,
    })
    table["review_verdict"] = load_review_verdicts(session, probe, table)

    # The rule, one column per criterion. Written as "passes" so a missing value counts as a fail.
    passes = pd.DataFrame({
        "contamination": table.rp_contamination < MAX_CONTAMINATION,
        "spikes": table.num_spikes > MIN_SPIKES,
        "snr_or_amplitude": (table.snr >= MIN_SNR) | (table.amplitude_median > MIN_AMPLITUDE_UV),
        "manual_review": table.review_verdict != "exclude",
    })
    passes_all = passes.all(axis=1).values
    # Strict check: the same rule with contamination at most 0.1
    strict_contamination = table.rp_contamination <= STRICT_CONTAMINATION
    passes_strict = (
        strict_contamination & passes.spikes & passes.snr_or_amplitude & passes.manual_review
    ).values
    # SNR-only check: the same rule with SNR at least 3 alone, without the amplitude clause
    passes_snr_only = (
        passes.contamination & passes.spikes & (table.snr >= MIN_SNR) & passes.manual_review
    ).values

    # Split neurons: keep one piece of each and mark the others as copies of it
    groups = group_split_neurons(same_neuron_pairs, len(table))
    same_neuron_as = [None] * len(table)
    for members in groups:
        keeper = choose_keeper(members, passes_all, passes_strict, num_spikes)
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
    # Strict check: good at the 0.1 cut. A copy is dropped only if the piece it copies passes it too.
    copies_strict_keeper = table.same_neuron_as.isin(table.unit_id[passes_strict])
    table["good_strict"] = passes_strict & ~copies_strict_keeper.values
    # SNR-only check: one piece per split neuron, chosen among the pieces that pass it. (The main
    # keeper can pass only through the amplitude clause while other pieces pass this check, so the
    # main keeper can't be reused here.)
    table["good_snr_only"] = one_piece_per_neuron(
        groups, passes_snr_only, passes_strict & passes_snr_only, num_spikes
    )
    return table


def plot_histograms(units, session):
    """Histograms of each measure with its cutoff in red, one figure per probe.

    The contamination panel also shows the strict 0.1 cut as a dotted line.
    """
    panels = [
        # (column, cutoff, bin edges)
        ("rp_contamination", MAX_CONTAMINATION, np.linspace(0, 1, 51)),
        ("num_spikes", MIN_SPIKES, np.logspace(1, 5, 50)),
        ("snr", MIN_SNR, np.linspace(0, 15, 50)),
        ("amplitude_median", MIN_AMPLITUDE_UV, np.linspace(0, 200, 50)),
        ("presence_ratio", LOW_PRESENCE, np.linspace(0, 1, 50)),
    ]
    for probe in get_session(session)["probes"]:
        probe_units = units[units.probe == probe]
        fig, axes = plt.subplots(1, len(panels), figsize=(20, 3.5))
        for ax, (column, cutoff, bins) in zip(axes, panels):
            # clip so values past the axis pile up at the edge instead of vanishing
            ax.hist(probe_units[column].clip(bins[0], bins[-1]), bins=bins, color="gray")
            ax.axvline(cutoff, color="red", linestyle="--")
            ax.set_title(column)
        axes[0].axvline(STRICT_CONTAMINATION, color="red", linestyle=":")
        axes[0].set_title("rp_contamination (dotted: strict check)")
        axes[1].set_xscale("log")
        axes[3].set_title("amplitude_median (uV)")
        fig.suptitle(
            f"{probe}: {probe_units.good.sum()} of {len(probe_units)} good "
            f"({probe_units.good_strict.sum()} at the strict 0.1 cut)"
        )
        fig.tight_layout()
        fig.savefig(f"outputs/metrics_{session}_{probe}.png", dpi=110)
        plt.close(fig)


def print_summary(units):
    """Per area: units, good units and main-analysis units (good and downward), the main analysis
    under the strict and SNR-only checks, and the manual review: units still waiting for a verdict
    and units it excluded."""
    for (probe, area), group in units.groupby(["probe", "area"]):
        good = group[group.good]
        main = group[group.main_analysis]
        strict = group[group.main_analysis_strict]
        snr_only = group[group.main_analysis_snr_only]
        pending = group[group.review_verdict == "pending"]
        excluded = group[group.review_verdict == "exclude"]
        print(
            f"{probe} {area:13}  units {len(group):3}  good {len(good):3}  main analysis {len(main):3}  "
            f"strict check {len(strict):3}  SNR only {len(snr_only):3}  "
            f"review: pending {len(pending):2}, excluded {len(excluded):2}"
        )


def check_areas_found(units, session):
    """Stops the run if an area listed for a probe in sessions.py holds no unit, which happens when
    its name doesn't match the probe's area-limits file."""
    for probe, areas in get_session(session)["probes"].items():
        found = sorted(units.area[units.probe == probe].unique())
        for area in areas:
            if area not in found:
                raise SystemExit(f"{session}: no unit on the {probe} probe is in area '{area}'; areas found: {found}")


def main(session):
    Path("outputs").mkdir(exist_ok=True)
    tables = []
    for probe in get_session(session)["probes"]:
        tables.append(build_unit_table(session, probe))
    units = pd.concat(tables, ignore_index=True)
    check_areas_found(units, session)
    units["low_presence"] = units.presence_ratio < LOW_PRESENCE
    # Upward spikes are likely axons, not cell bodies, and trough-based features
    # don't fit them. They stay in the table but out of the main analysis.
    units["main_analysis"] = units.good & (units.polarity == "negative")
    units["main_analysis_strict"] = units.good_strict & (units.polarity == "negative")
    units["main_analysis_snr_only"] = units.good_snr_only & (units.polarity == "negative")
    units.to_csv(f"outputs/good_cells_{session}.csv", index=False)
    plot_histograms(units, session)
    print_summary(units)


if __name__ == "__main__":
    main(session_from_command_line())
