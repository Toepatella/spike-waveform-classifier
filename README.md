# Spike waveform classifier

Data lives outside the repo in `C:\Users\tonyt\spike-data` (set `SPIKE_DATA_DIR`
to use another folder). See `DATA_DIR` in `load_data.py`.

1. `uv run good_cells.py`: Task 0. Picks good cells, saves `outputs/good_cells_<session>.csv` and metric histograms.
2. `uv run plot_waveforms.py`: Task 1. Sparse waveform plots of good and rejected units.

`outputs/` is ignored by git; rerun the scripts to recreate it.
