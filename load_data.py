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
    # Allow it only while loading Juan's files. read_only keeps his files unchanged.
    numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True
    try:
        return si.load_sorting_analyzer(
            DATA_DIR / session / probe / "Postcuration_Analyzer.zarr", read_only=True
        )
    finally:
        numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = False


def load_metrics(session, probe):
    # Use the CSV: the analyzer's saved copy is missing SNR for HPC
    return pd.read_csv(DATA_DIR / session / probe / "quality_metrics.csv", index_col=0)


class PlainDataUnpickler(pickle.Unpickler):
    # Reads plain data only (dicts, strings, numbers). Refuses to import any class, so it can't run code.
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")


def load_area_limits(session, probe):
    # e.g. {"PFC": 3650, "White Matter": 2350, "OFC": 1200}
    with open(DATA_DIR / session / probe / "Anatomical_limits.pkl", "rb") as f:
        return PlainDataUnpickler(f).load()


def area_of(depth_um, limits):
    # Assumes each limit is the TOP of that area, in probe coordinates (um, same frame as Phy's
    # depth; the physical tip is a few hundred um below 0). The only reading that puts PFC, CA3
    # and DG on the probe. Confirm with Juan.
    for area, top in sorted(limits.items(), key=lambda item: item[1]):
        if depth_um <= top:
            return area
    return "above"
