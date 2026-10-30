"""Load the sorted recordings from Juan (SpikeInterface 0.105 SortingAnalyzer folders).

Parts of these folders (the preprocessing recipe and saved PCA models) are stored
in Python's pickle format. Reading pickle can run code, so numcodecs blocks it by
default. We allow it only while loading these trusted lab files, then turn the
block back on.
"""

import pickle
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


def load_analyzer(session: str, probe: str):
    """Load one probe's SortingAnalyzer, e.g. load_analyzer("2026-07-21", "PFC")."""
    folder = DATA_DIR / session / probe / "Postcuration_Analyzer.zarr"
    with _allow_pickle():
        # Loads all saved extensions now, so nothing pickled is read later.
        return si.load_sorting_analyzer(folder, load_extensions=True)


class _PlainDataUnpickler(pickle.Unpickler):
    # Only plain data (dicts, strings, numbers) is allowed: any attempt to
    # import a class or function is refused, so loading can't run code.
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name}")


def load_area_limits(session: str, probe: str) -> dict[str, float]:
    """Read Juan's Anatomical_limits.pkl, e.g. {"PFC": 3650, "White Matter": 2350, "OFC": 1200}.

    Values are distances in micrometres from the probe tip.
    """
    path = DATA_DIR / session / probe / "Anatomical_limits.pkl"
    with open(path, "rb") as f:
        return dict(_PlainDataUnpickler(f).load())


def assign_area(depth_um: float, limits: dict[str, float]) -> str:
    """Label a position on the probe with its brain area.

    Assumption (to confirm with Juan): each value in the limits file is the
    TOP edge of that area, counted up from the probe tip. So for the PFC probe,
    OFC is 0-1200 um, white matter 1200-2350 um and PFC 2350-3650 um. Anything
    above the highest edge is labelled "above".
    """
    for area, top in sorted(limits.items(), key=lambda item: item[1]):
        if depth_um <= top:
            return area
    return "above"
