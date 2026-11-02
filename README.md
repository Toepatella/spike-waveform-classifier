# Spike waveform classifier

Data lives outside the repo in `C:\Users\tonyt\spike-data` (set `SPIKE_DATA_DIR`
to use another folder). See `DATA_DIR` in `load_data.py`.

Run everything with `uv run main.py`, or run the steps one at a time:

1. `uv run good_cells.py`: Task 0. Picks good cells, saves `outputs/good_cells_<session>.csv` and metric histograms.
2. `uv run plot_waveforms.py`: Task 1. Sparse waveform plots of good, rejected and flagged units.

`outputs/` is ignored by git; rerun the scripts to recreate it.

Each script keeps its work in a `main()` function; `main.py` imports the scripts and calls
those functions in order.

## Good-cell rule

A good cell should give a clean average waveform from one neuron:

- contamination (Juan's `rp_contamination`, Llobet et al. 2022) at most 10%;
- more than 500 spikes (Juan's average waveforms use up to 500 random spikes);
- SNR at least 3, or median amplitude above 30 µV, so big units on noisy channels still count;
- not a split-off copy of another unit (near-identical template, and the two never fire
  within 1 ms of each other).

The spike and size cuts follow Juan's advice (2026-11-02). He suggested 0.5 for
contamination, but units that pass only at 0.5 look like mixed clusters on his waveform
isolation scores (`nn_hit_rate`, silhouette). So 10% stays the main cut, and 0.5 is a
robustness check.

Juan's score assumes a steady firing rate, so bursty cells (mostly narrow-spiking) that
fail it are kept anyway if their autocorrelogram shows a clean refractory gap
(`clean_acg`). These units are marked in `rescued_by_acg`.

Positive-going (upward) spikes are likely axonal (Juan agrees), so they stay in the table
but are left out of the main analysis. Low-presence units are kept but flagged for a look.

The main analysis and two robustness checks are columns in
`outputs/good_cells_<session>.csv`:

| Version | Units | How to select them |
|---|---|---|
| Main | good and downward | `main_analysis` |
| Check 1 | as main, but contamination below 0.5 (Juan's cut) | `main_analysis_loose` |
| Check 2 | as main, without the units kept by the autocorrelogram check | `main_analysis & ~rescued_by_acg` |

The main version was chosen before any feature results existed. Report results for all
three versions.

Figure titles show two IDs. `unit` is SpikeInterface's unit ID and `Phy` is the cluster ID
in Juan's Phy files. They differ for about half the units, so use Phy IDs with Juan.

## Data provenance

Juan's pipeline, as recorded in the analyzer files:

- Units: Phy "good" clusters only (noise and MUA excluded). Spikes closer than 0.2 ms within a unit were removed.
- Probes: Neuropixels 1.0 NHP at 30 kHz, with the probe's built-in 300 Hz filter off (`Broadband_rec.zarr`).
- Before waveforms: ±5 ms around 640 events replaced by cubic interpolation; 300 Hz 2nd-order Butterworth high-pass, run forward and backward (zero-phase); global median reference; motion correction (kriging). No whitening.
- Waveforms: up to 500 random spikes per unit, 2 ms either side of the peak.
