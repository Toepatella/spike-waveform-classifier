"""Load the sorted recordings from Juan (SpikeInterface 0.105 SortingAnalyzer folders).

Parts of these folders (the preprocessing recipe and saved PCA models) are stored
in Python's pickle format. Reading pickle can run code, so numcodecs blocks it by
default. We allow it only while loading these trusted lab files, then turn the
block back on.
"""

from contextlib import contextmanager

import numcodecs.pickles
import spikeinterface as si

from paths import DATA_DIR


@contextmanager
def _allow_pickle():
    previous = numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE
    numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True
    try:
        yield
    finally:
        numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = previous


def load_analyzer(session: str, region: str):
    """Load one region's SortingAnalyzer, e.g. load_analyzer("2026-07-21", "PFC")."""
    folder = DATA_DIR / session / region / "Postcuration_Analyzer.zarr"
    with _allow_pickle():
        # Loads all saved extensions now, so nothing pickled is read later.
        return si.load_sorting_analyzer(folder, load_extensions=True)
