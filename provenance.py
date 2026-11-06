"""Record how each probe's data were processed, from the files saved with the recording."""

from pathlib import Path

import numpy as np
import spikeinterface as si

from load_data import load_analyzer, load_processing_chain
from sessions import get_session, session_from_command_line

MAX_TEXT = 100  # longer text is shown by its length, longer objects are cut to this many characters
MAX_LIST_VALUES = 4  # lists holding more values are shown by their count
STORED_RESULTS = ["random_spikes", "waveforms", "templates"]  # analyzer extensions whose settings are listed


def count_values(value):
    """How many numbers or strings a value holds, counting inside nested lists and arrays."""
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            total += count_values(item)  # calls itself for lists inside lists
        return total
    return 1


def describe_setting(value):
    """Short text for one setting: the value itself if it is short, otherwise its size."""
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, str):
        if len(value) <= MAX_TEXT:
            return value
        return f"<text of {len(value)} characters>"
    if isinstance(value, (list, tuple, np.ndarray)):
        number = count_values(value)
        if number > MAX_LIST_VALUES:
            return f"<{number} values>"
        if isinstance(value, np.ndarray):
            return str(value.tolist())
        return str(list(value))
    text = str(value)  # objects such as the motion estimate describe themselves
    if len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + "..."


def describe_step(name, settings):
    """One line for one processing step: its name, then each setting."""
    parts = []
    for key, value in settings.items():
        parts.append(f"{key}={describe_setting(value)}")
    line = f"{name}: " + ", ".join(parts)
    if name == "HighpassFilterRecording" and "direction" not in settings:
        line += f" (direction not saved; the SpikeInterface {si.__version__} default is forward-backward)"
    return line


def describe_probe(session, probe):
    """Lines about one probe: the recording's size, every processing step, and how the stored
    waveforms, templates and metrics were made."""
    analyzer = load_analyzer(session, probe)
    fs = analyzer.sampling_frequency
    lines = [
        f"== {probe}",
        f"sampling rate (Hz): {fs}",
        f"duration (s): {analyzer.get_total_samples() / fs:.3f}",
        f"channels: {analyzer.get_num_channels()}",
        "processing, first step first:",
    ]
    for name, settings in load_processing_chain(session, probe, analyzer):
        lines.append("  " + describe_step(name, settings))
    lines.append("stored results:")
    for name in STORED_RESULTS:
        lines.append(f"  {name}: {analyzer.get_extension(name).params}")
    metric_names = analyzer.get_extension("quality_metrics").params["metric_names"]
    lines.append("  quality_metrics: " + ", ".join(metric_names))
    return lines


def main(session):
    Path("outputs").mkdir(exist_ok=True)
    lines = [f"session: {session}"]
    for probe in get_session(session)["probes"]:
        lines.append("")
        lines = lines + describe_probe(session, probe)
    path = Path("outputs") / f"processing_{session}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Processing record: {path}")


if __name__ == "__main__":
    main(session_from_command_line())
