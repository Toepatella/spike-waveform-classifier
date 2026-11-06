"""Facts about each recorded session that the code doesn't read from the data files."""

import sys

# One entry per session, named like its folder in the data directory. "probes" maps each probe
# folder to the areas analysed on it; the area names must match the probe's area-limits file.
# animal, insertion and track are kept for pooling sessions later; no step reads them yet.
SESSIONS = {
    "2026-07-21": {
        "animal": "Mars",  # from the SpikeGLX folder name in the recording's provenance
        "insertion": "unknown",  # acute or chronic: not recorded in the files
        "track": "unknown",  # penetration track: not recorded in the files
        "probes": {
            "PFC": ["PFC"],
            "HPC": ["CA3", "DG"],
        },
    },
}


def get_session(name):
    """The entry for one session. Stops the run, listing the known sessions, if the name is unknown."""
    if name not in SESSIONS:
        known = ", ".join(sorted(SESSIONS))
        raise SystemExit(f"Unknown session '{name}'. Known sessions: {known}")
    return SESSIONS[name]


def session_from_command_line():
    """The session named as the only command-line argument, checked against SESSIONS."""
    if len(sys.argv) != 2:  # sys.argv[0] is the script's own name, so one argument makes 2
        known = ", ".join(sorted(SESSIONS))
        raise SystemExit(f"Name one session. Known sessions: {known}")
    name = sys.argv[1]
    get_session(name)  # stops the run if the name is unknown
    return name
