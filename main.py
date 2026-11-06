"""Run every step, in order, for one session."""

import features
import firing
import good_cells
import plot_waveforms
import provenance
import report
from sessions import session_from_command_line


def main(session):
    print(f"--- Processing steps recorded for {session} ---")
    provenance.main(session)
    print("--- Task 0: picking good cells ---")
    good_cells.main(session)
    print("--- Task 1: plotting waveforms ---")
    plot_waveforms.main(session)
    print("--- Task 2: waveform features ---")
    features.main(session)
    print("--- Firing rate and LvR ---")
    firing.main(session)
    print("--- Task 2: narrow shares ---")
    report.main(session)
    print("--- Done. Results are in outputs/ ---")


if __name__ == "__main__":
    main(session_from_command_line())
