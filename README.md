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

A good cell should give a clean average waveform from one neuron: contamination
(Juan's `rp_contamination`) at most 10%, at least 200 spikes, SNR at least 3, and not a
split-off copy of another unit (near-identical template, and the two never fire within 1 ms
of each other). Juan's score assumes a steady firing rate, so bursty cells (mostly
narrow-spiking) that fail it are kept anyway if their autocorrelogram shows a clean
refractory gap. They are marked in `rescued_by_acg`, so results can be checked without them.
Positive-going spikes are kept but marked in `polarity`. Low-presence units are kept but
flagged for a look.

## Data provenance

Juan's pipeline, as recorded in the analyzer files:

- Units: Phy "good" clusters only (noise and MUA excluded). Spikes closer than 0.2 ms within a unit were removed.
- Probes: Neuropixels 1.0 NHP at 30 kHz, with the probe's built-in 300 Hz filter off (`Broadband_rec.zarr`).
- Before waveforms: ±5 ms around 640 events replaced by cubic interpolation; 300 Hz 2nd-order Butterworth high-pass, run forward and backward (zero-phase); global median reference; motion correction (kriging). No whitening.
- Waveforms: up to 500 random spikes per unit, 2 ms either side of the peak.
