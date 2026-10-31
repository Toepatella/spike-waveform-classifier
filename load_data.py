"""Load Juan's sorted recordings."""

import os
import pickle
from pathlib import Path

import numcodecs.pickles
import pandas as pd
import spikeinterface as si

# Data lives outside the repo (too big for git/OneDrive)
DATA_DIR = Path(os.environ.get("SPIKE_DATA_DIR", r"C:\Users\tonyt\spike-data"))


def load_analyzer(session, probe):
    # Parts of the folder are pickled, which numcodecs blocks by default.
    # Allow it only while loading Juan's files.
    numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True
    try:
        return si.load_sorting_analyzer(DATA_DIR / session / probe / "Postcuration_Analyzer.zarr")
    finally:
        numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = False


def load_metrics(session, probe):
    # Use the CSV: the analyzer's saved copy is missing SNR for HPC
    return pd.read_csv(DATA_DIR / session / probe / "quality_metrics.csv", index_col=0)


def load_area_limits(session, probe):
    # e.g. {"PFC": 3650, "White Matter": 2350, "OFC": 1200}
    with open(DATA_DIR / session / probe / "Anatomical_limits.pkl", "rb") as f:
        return pickle.load(f)


def area_of(depth_um, limits):
    # Assumes each limit is the TOP of that area, in um from the probe tip (confirm with Juan)
    for area, top in sorted(limits.items(), key=lambda item: item[1]):
        if depth_um <= top:
            return area
    return "above"
