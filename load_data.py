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
    """Open one probe's SortingAnalyzer, e.g. load_analyzer("2026-07-21", "PFC").

    Parts of the folder are pickled, which numcodecs blocks by default because pickle can run
    code. The block is lifted only while loading Juan's trusted files. read_only keeps his
    files unchanged.

    load_extensions=False: each saved result (templates, correlograms, ...) loads the first time
    get_extension asks for it. This skips Juan's saved PCA models, which we never use; they are
    pickled scikit-learn objects and warn because he saved them with an older scikit-learn.
    """
    numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True
    try:
        return si.load_sorting_analyzer(
            DATA_DIR / session / probe / "Postcuration_Analyzer.zarr", read_only=True, load_extensions=False
        )
    finally:
        numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = False


def load_metrics(session, probe):
    """Juan's quality metrics, one row per unit (indexed by unit ID).

    Uses his CSV rather than the analyzer's saved copy, which is missing SNR for HPC.
    """
    return pd.read_csv(DATA_DIR / session / probe / "quality_metrics.csv", index_col=0)


class PlainDataUnpickler(pickle.Unpickler):
    """Reads plain data only (dicts, strings, numbers).

    Refuses to import any class, so loading can't run code. This is the approach the Python
    docs recommend for restricting pickle ("Restricting Globals").
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")


def load_area_limits(session, probe):
    """Juan's area boundaries, e.g. {"PFC": 3650, "White Matter": 2350, "OFC": 1200}."""
    with open(DATA_DIR / session / probe / "Anatomical_limits.pkl", "rb") as f:
        return PlainDataUnpickler(f).load()


def area_of(depth_um, limits):
    """The brain area at this position on the probe.

    Assumes each limit is the TOP of that area, in probe coordinates (um, same frame as Phy's
    depth; the physical tip is a few hundred um below 0). It's the only reading that puts PFC,
    CA3 and DG on the probe. Confirm with Juan.
    """
    for area, top in sorted(limits.items(), key=lambda item: item[1]):  # lowest limit first
        if depth_um <= top:
            return area
    return "above"
