# Spike waveform classifier

Data live outside the repo, in `C:\Users\tonyt\spike-data` or the folder named by the
`SPIKE_DATA_DIR` environment variable (`DATA_DIR` in `load_data.py`). Each session is a folder
named as in `sessions.py`, with one subfolder per probe holding `Postcuration_Analyzer.zarr`,
`quality_metrics.csv`, `Anatomical_limits.pkl` and `Broadband_rec.zarr`.

`uv run main.py <session>` runs every step for one session. Each step reads the outputs of the
steps before it, and also runs on its own with the same argument:

1. `uv run provenance.py <session>`: every processing step recorded in the data files, saved as `outputs/processing_<session>.txt`.
2. `uv run good_cells.py <session>`: Task 0. Picks good cells, saves `outputs/good_cells_<session>.csv` and metric histograms.
3. `uv run plot_waveforms.py <session>`: Task 1. Sparse waveform plots of good, rejected and flagged units, plus review sheets for the units listed for manual review.
4. `uv run features.py <session>`: Task 2. Waveform features of every unit, saved as `outputs/features_<session>.csv`, with waveform vectors and QC figures.
5. `uv run firing.py <session>`: firing rate and LvR outside the artifact windows, saved as `outputs/firing_<session>.csv`.
6. `uv run report.py <session>`: Task 2. Narrow shares per area for every version, saved as `outputs/narrow_shares_<session>.csv`.

`uv run checks.py <session>` checks the outputs and the measurement code, printing PASS, FAIL or
SKIP for each check. Rules fixed before later analyses are in `analysis_plan.md`.

`outputs/` is not tracked by git. The scripts recreate everything in it except
`outputs/reference_0cb6002/`: copies of the unit table and metric histograms made with commit
0cb6002, which `checks.py` compares the current unit table against.

## Good-cell rule

A good cell should give a clean average waveform from one neuron:

- contamination (`rp_contamination`, Llobet et al. 2022) below 0.5;
- more than 500 spikes (the average waveforms use up to 500 random spikes);
- SNR at least 3, or median amplitude above 30 µV, so big units on noisy channels still count;
- not a split-off copy of another unit (near-identical template, and the two never fire
  within 1 ms of each other).

These cuts were recommended with the sorted data. `rp_contamination` compares a unit's spike
pairs 0.2–1 ms apart (`rp_violations`; closer pairs were removed as duplicates) with the number
expected if spikes from other neurons arrived at a steady rate, independent of the unit. When
they rise and fall with the unit's own firing, the score reads high. The expected number grows
with the square of the firing rate, so a unit below about 0.7 Hz passes only with no such pairs.

`rp_contamination_lower` is the lower end of a 95% confidence interval for `rp_contamination`:
the lowest Poisson mean of `rp_violations` consistent with the count (Garwood's exact interval),
put through the same formula. It is not part of the rule; the narrow-share report and the
intermediate-class check in `analysis_plan.md` use it. A unit with `rp_contamination` below 0.5 can
have `rp_contamination_lower` above 0.20 only with many spikes: about 7,600 or more (2.4 Hz) in
the 52-minute 2026-07-21 session.

`acg_gap` (pairs under 1 ms apart relative to 10–25 ms apart, from the autocorrelogram) and
`nn_hit_rate` (separation from neighbouring units in PCA space, which falls with spike size) are
in the table but not used in the rule.

Positive-going (upward) spikes are likely axonal, so they stay in the table but are left out of
the main analysis. Units with `presence_ratio` below 0.5 (`low_presence`: silent for long
stretches, not just slow) are kept; no version uses the flag.

`near_wm_limit` marks units outside the white matter within 100 µm above its top. The area limits
are estimates, so these may be white-matter units. It is not part of the rule.

The main analysis and two checks are columns in `outputs/good_cells_<session>.csv`:

| Version | Units | Column |
|---|---|---|
| Main | good and downward | `main_analysis` |
| Strict check | as main, but contamination at most 0.1 | `main_analysis_strict` |
| SNR-only check | as main, but SNR at least 3, without the 30 µV clause | `main_analysis_snr_only` |

Each version keeps one piece of each split neuron. The main version was fixed before the
waveform-feature analysis. Results are reported for every version, and for the main version
without the likely-contaminated, near-white-matter or Bombcell-flagged units.

Figure titles show two IDs. `unit` is SpikeInterface's unit ID and `Phy` is the cluster ID
in the Phy files. They differ for about half the units.

## Waveform features

`features.py` measures each downward unit's stored average template (the mean of up to 500
random spikes) on its channel with the largest |value|. The template is first interpolated ×10,
assuming it holds no frequencies above half its sampling rate: SciPy's `resample_poly(trace, 10, 1,
window=("kaiser", 10))`, a 201-tap filter spanning 10 original samples on each side, with zeros
beyond the template's ends. The trough is sought around the sorted spike time (the time the
sorter gave the spike, 2.0 ms into each stored template); the other times are measured from the
trough.

- trough: the minimum within 0.3 ms of the sorted spike time;
- baseline: the median of the template's first 0.5 ms. The trace is scaled so the baseline is 0
  and the trough −1;
- post-trough peak: the maximum 0.1–1.5 ms after the trough. Its time is the centre (mean time) of
  the samples in that window at or above 90% of the maximum;
- `trough_to_peak_ms`, and `peak_trough_ratio` (the peak's height over the trough's depth);
- `peak_decay_ms`: from the peak time to the first fall to half the peak's height after its top;
- `repolarization_ms`: from the peak time to the steepest descent after the peak's top, before
  the trace first returns to the baseline;
- `half_width_ms` (width at half the trough depth), and `falling_phase_ms` and `rising_phase_ms`
  (from and to 10% of the trough depth);
- slopes in trough depths per ms: `fall_slope` (steepest descent in the 0.5 ms before the trough),
  `rise_slope` (steepest rise between the trough and the peak) and `recovery_slope` (the steepest
  descent where `repolarization_ms` ends);
- `pre_peak_ratio` and `pre_peak_width_ms`: the largest value in the 1 ms before the trough, and
  its width at half its height.

A landmark that isn't found leaves its features empty and names the reason in `missing`; nothing
is filled in. The exception is `pre_peak_ratio`, which is kept when the trace stays at or below
the baseline before the trough; only `pre_peak_width_ms` is then empty. The codes are `no_trough`,
`no_half_width`, `no_fall_start`, `no_rise_end`, `no_pre_peak`, `no_positive_peak`,
`peak_at_window_start` and `peak_at_window_end` (the peak's top reaches 0.1 or 1.5 ms after the
trough, so part of it may lie outside the search), `decay_outside_window`, `no_falling_branch` and
`positive_spike` (upward units, not measured). `channel` is the measured
channel's position in the analyzer's channel list.

`bombcell_non_somatic` marks the non-somatic shape defined by Bombcell (a unit-curation toolbox for
Neuropixels data), with its thresholds converted from samples to ms: a pre-trough peak narrower
than 0.15 ms, at least 0.2 of the trough and 3 times the post-trough peak, before a trough with a
half-width under 0.2 ms.

`outputs/waveform_vectors_<session>.npz` holds each unit's scaled trace from 0.5 ms before to
1.5 ms after the trough, on a 10 µs grid (rows of NaN for upward units).
`outputs/features_<session>_<probe>_<area>.png` shows the trough-to-peak distribution of the
main-analysis units and 12 of their traces with the landmarks marked.

## Firing statistics

`firing.py` leaves out the spikes inside the stretches replaced around artifact events (in
2026-07-21: 640 events, ±5 ms each) and the time those stretches cover. `spikes_in_windows` counts
the spikes left out and `time_outside_windows_s` the time outside the stretches. `rate_hz` is the
remaining spikes over that time. `lvr` is the local variation with refractoriness (Shinomoto et al.
2009) over pairs of consecutive intervals, leaving out intervals that span a replaced stretch: near
1 for random firing, lower for regular firing and higher for bursty firing; R = 5 ms is its
refractoriness constant. `lvr_pairs` is the number of interval pairs. The unit table's
`firing_rate` covers the whole session. `outputs/firing_<session>.png` plots LvR against rate for
the main-analysis units.

## Narrow shares

`report.py` gives the share of narrow units (trough-to-peak below 0.40 ms, provisional) per probe
and area, among units with a trough-to-peak, for six versions: the main analysis, the strict
check, the SNR-only check, and the main analysis without the units with `rp_contamination_lower`
above 0.20, without the `near_wm_limit` units, and without the `bombcell_non_somatic` units.
Columns: `units`, `no_ttp` (units without a trough-to-peak), `narrow`, and `narrow_share` (narrow
over units with a trough-to-peak).

## Manual review

`curation/manual_review_<session>.csv` holds verdicts on listed units. There is no routine review;
the file records one-off exclusions. The 2026-07-21 list holds the units that an earlier rule
(contamination at most 0.1, or a clean autocorrelogram) kept only because of their autocorrelogram.
Its 20 `keep` verdicts apply the rule that units below 0.5 are good, without a check by eye; the 5
units at 0.5 or above have no verdict and show as `pending`.

Verdicts (`keep`, `exclude` or `unsure`, with an optional `note`) go in the `verdict` column and
take effect on the next run. How verdicts are used was fixed before the review:

- `exclude` removes a unit from every version (the main analysis and every check).
- `keep` and `unsure` leave it in.
- A review can never add a unit.

The file uses Phy IDs, and each one is checked against its unit ID, so a mixed-up ID
stops the run instead of excluding the wrong unit. `outputs/review_<session>_<area>.png`
shows each listed unit's waveforms and autocorrelogram.

## Data provenance

Sorting and preprocessing of the 2026-07-21 session, as recorded in the analyzer and broadband
recording files (listed in full by `provenance.py`):

- Units: Phy "good" clusters only (noise and MUA excluded). Spikes closer than 0.2 ms within a unit were removed.
- Probes: Neuropixels 1.0 NHP at 30 kHz, with the probe's built-in 300 Hz filter off (`Broadband_rec.zarr`, saved after the phase shift, centring, channel selection and artifact interpolation below).
- Before waveforms: ADC phase-shift correction; each channel centred on its median; channel selection; ±5 ms around 640 events replaced by cubic interpolation; 300 Hz 2nd-order Butterworth high-pass (its direction isn't saved; SpikeInterface's default runs it forward and backward, zero-phase); global median reference; motion correction (kriging). No whitening.
- Waveforms: up to 500 random spikes per unit, from 2 ms before to 2 ms after the sorted spike time.
- Areas: estimated upper boundary of each area on the probe (`Anatomical_limits.pkl`). "PFC" is area 8/46.
