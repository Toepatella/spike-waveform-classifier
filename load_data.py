"""Load the sorted recordings."""

import json
import os
import pickle
from pathlib import Path

import numcodecs.pickles
import numpy as np
import pandas as pd
import spikeinterface as si

# Data lives outside the repo (too big for git/OneDrive)
DATA_DIR = Path(os.environ.get("SPIKE_DATA_DIR", r"C:\Users\tonyt\spike-data"))


def load_analyzer(session, probe):
    """Open one probe's SortingAnalyzer, e.g. load_analyzer(session, "PFC").

    Parts of the folder are pickled, which numcodecs blocks by default because pickle can run
    code. The block is lifted only while loading the trusted sorted files. read_only keeps
    them unchanged.

    load_extensions=False: each saved result (templates, correlograms, ...) loads the first time
    get_extension asks for it. This skips the saved PCA models, which are never used; they are
    pickled scikit-learn objects and warn because they were saved with an older scikit-learn.
    """
    numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = True
    try:
        return si.load_sorting_analyzer(
            DATA_DIR / session / probe / "Postcuration_Analyzer.zarr", read_only=True, load_extensions=False
        )
    finally:
        numcodecs.pickles.ALLOW_UNSAFE_PICKLE_DECODE = False


def load_metrics(session, probe):
    """Quality metrics from the sorting, one row per unit (indexed by unit ID).

    Uses the CSV rather than the analyzer's saved copy, which is missing SNR for HPC.
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
    """Area boundaries, e.g. {"PFC": 3650, "White Matter": 2350, "OFC": 1200}."""
    with open(DATA_DIR / session / probe / "Anatomical_limits.pkl", "rb") as f:
        return PlainDataUnpickler(f).load()


def area_of(depth_um, limits):
    """The brain area at this position on the probe.

    Assumes each limit is the TOP of that area, in probe coordinates (um, same frame as Phy's
    depth; the physical tip is a few hundred um below 0). It's the only reading that puts PFC,
    CA3 and DG on the probe, and it matches the area notes sent with the data.
    """
    for area, top in sorted(limits.items(), key=lambda item: item[1]):  # lowest limit first
        if depth_um <= top:
            return area
    return "above"


def load_broadband_provenance(session, probe):
    """How the stored broadband recording was made: the SpikeInterface provenance saved with it,
    a nested dict that starts at the last step."""
    path = DATA_DIR / session / probe / "Broadband_rec.zarr" / ".zattrs"
    with open(path, encoding="utf-8") as f:
        attributes = json.load(f)
    return attributes["provenance"]


def chain_steps(recording_dict):
    """The steps of a saved SpikeInterface recording, first step first.

    recording_dict is nested: each step holds the step before it under "recording" (or
    "parent_recording"). Returns [(class name, settings), ...], with that nested step left out of
    the settings.
    """
    steps = []
    node = recording_dict
    while isinstance(node, dict) and "class" in node:
        kwargs = node.get("kwargs", {})
        settings = {}
        for key, value in kwargs.items():
            if key not in ("recording", "parent_recording"):
                settings[key] = value
        name = node["class"].split(".")[-1]  # e.g. "spikeinterface....HighpassFilterRecording" -> "HighpassFilterRecording"
        steps.append((name, settings))
        node = kwargs.get("recording", kwargs.get("parent_recording"))  # the step before this one
    steps.reverse()
    return steps


def load_processing_chain(session, probe, analyzer):
    """Every processing step from the raw SpikeGLX file to the recording the waveforms come from,
    first step first: the steps saved with the broadband recording, then the analyzer's own."""
    steps = chain_steps(load_broadband_provenance(session, probe))
    steps = steps + chain_steps(analyzer.recording.to_dict(recursive=True))
    return steps


def load_artifact_windows(session, probe, fs, total_samples):
    """The stretches of the recording replaced by interpolation around artifact events, as
    (starts, stops) in samples, in time order, both ends included and kept inside the recording.
    Empty if no such step was recorded.

    The window sizes follow SpikeInterface's remove_artifacts: int(ms * fs / 1000) samples on each
    side of the event.
    """
    for name, settings in chain_steps(load_broadband_provenance(session, probe)):
        if name == "RemoveArtifactsRecording":
            # list_triggers holds one list of event samples per recording segment; these recordings have one
            events = np.sort(np.array(settings["list_triggers"][0], dtype=np.int64))
            before = int(settings["ms_before"] * fs / 1000)
            after = int(settings["ms_after"] * fs / 1000)
            starts = np.maximum(events - before, 0)
            stops = np.minimum(events + after, total_samples - 1)
            return starts, stops
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
