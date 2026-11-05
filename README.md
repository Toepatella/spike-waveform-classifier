# Spike waveform classifier

Data lives outside the repo in `C:\Users\tonyt\spike-data` (set `SPIKE_DATA_DIR`
to use another folder). See `DATA_DIR` in `load_data.py`.

Run everything with `uv run main.py`, or run the steps one at a time:

1. `uv run good_cells.py`: Task 0. Picks good cells, saves `outputs/good_cells_<session>.csv` and metric histograms.
2. `uv run plot_waveforms.py`: Task 1. Sparse waveform plots of good, rejected and flagged units, plus review sheets for the units listed for manual review.

`outputs/` is not tracked by git; the scripts recreate it.

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

`acg_gap` (pairs under 1 ms apart relative to 10–25 ms apart, from the autocorrelogram) and
`nn_hit_rate` (separation from neighbouring units in PCA space, which falls with spike size) are
in the table but not used in the rule.

Positive-going (upward) spikes are likely axonal, so they stay in the table
but are left out of the main analysis. Low-presence units are kept but flagged for a look.

The main analysis and a strict check are columns in `outputs/good_cells_<session>.csv`:

| Version | Units | Column |
|---|---|---|
| Main | good and downward | `main_analysis` |
| Strict check | as main, but contamination at most 0.1 | `main_analysis_strict` |

The main version was fixed before the waveform-feature analysis. Results are reported for
both versions.

Figure titles show two IDs. `unit` is SpikeInterface's unit ID and `Phy` is the cluster ID
in the Phy files. They differ for about half the units.

## Manual review

`curation/manual_review_<session>.csv` lists units checked by eye in Phy. The 2026-07-21 list
holds the units that an earlier rule (contamination at most 0.1, or a clean autocorrelogram)
kept only because of their autocorrelogram.

To record a review:

1. In the `verdict` column, enter `keep`, `exclude` or `unsure` for each unit, with an
   optional `note`.
2. Rerun `uv run main.py`.

How verdicts are used was fixed before the review:

- `exclude` removes a unit from every version (main and strict check).
- `keep` and `unsure` leave it in.
- A review can never add a unit.

The file uses Phy IDs, and each one is checked against its unit ID, so a mixed-up ID
stops the run instead of excluding the wrong unit. `outputs/review_<session>_<area>.png`
shows each listed unit's waveforms and autocorrelogram for the review.

## Data provenance

Sorting and preprocessing, as recorded in the analyzer files:

- Units: Phy "good" clusters only (noise and MUA excluded). Spikes closer than 0.2 ms within a unit were removed.
- Probes: Neuropixels 1.0 NHP at 30 kHz, with the probe's built-in 300 Hz filter off (`Broadband_rec.zarr`).
- Before waveforms: ±5 ms around 640 events replaced by cubic interpolation; 300 Hz 2nd-order Butterworth high-pass, run forward and backward (zero-phase); global median reference; motion correction (kriging). No whitening.
- Waveforms: up to 500 random spikes per unit, 2 ms either side of the peak.
- Areas: estimated upper boundary of each area on the probe (`Anatomical_limits.pkl`). "PFC" is area 8/46.
