"""Run every step in order: uv run main.py"""

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
