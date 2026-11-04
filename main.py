"""Run Task 0 and Task 1 in order."""

import good_cells
import plot_waveforms


def main():
    print("--- Task 0: picking good cells ---")
    good_cells.main()
    print("--- Task 1: plotting waveforms ---")
    plot_waveforms.main()
    print("--- Done. Results are in outputs/ ---")


if __name__ == "__main__":
    main()
