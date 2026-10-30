# Spike waveform classifier

## Data

Recordings are not stored in this repository. They live in
`C:\Users\tonyt\spike-data`, one folder per session (for example `2026-07-21/PFC`
and `2026-07-21/HPC`). Code finds them through `DATA_DIR` in `paths.py`. To use
another location, set the `SPIKE_DATA_DIR` environment variable.

## Steps

1. `uv run good_cells.py`: Task 0. Applies the good-cell rule (set at the top of
   the file), labels each unit's brain area, and saves `outputs/good_cells_<session>.csv`
   plus metric histograms and depth plots.
2. `uv run plot_waveforms.py`: Task 1. Sparse waveform plots of good, rejected,
   low-presence and possible-duplicate units for PFC, CA3 and DG.

Figures and tables go to `outputs/`, which git ignores; rerun the scripts to
recreate them.
