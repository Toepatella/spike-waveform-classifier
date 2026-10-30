"""Task 1: sparse waveform plots.

Run with:  uv run plot_waveforms.py
(Run good_cells.py first: this script reads the good-cell table it saves.)

Each panel shows one unit on the recording sites near it ("sparse" = only
those sites, not all 384). Every trace sits at its site's depth on the probe
(y-axis, in um). Faint lines are individual spikes; the bold line is their
average (the template). The x-axis is time in samples around the spike
(30 samples = 1 ms).

For each brain area this saves:
  - a random sample of good units,
  - rejected units, one per failed criterion, to show what each cutoff removes,
  - good units flagged for low presence,
  - possible duplicate pairs side by side.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.widgets as sw

from good_cells import OUTPUT_DIR, SESSION
from load_data import load_analyzer

# Which areas to plot on each probe (the ones the project uses).
AREAS_TO_PLOT = {"PFC": ["PFC"], "HPC": ["CA3", "DG"]}

NUM_GOOD_EXAMPLES = 8
NUM_FLAGGED_EXAMPLES = 8
NUM_DUPLICATE_PAIRS = 4

# Spikes drawn per unit. More looks busier and is slower to draw.
SPIKES_PER_UNIT = 50

# Fixed seed so the random sample of units is the same every run.
RANDOM_SEED = 0


def plot_units(analyzer, units: pd.DataFrame, title: str, path: Path, ncols: int = 4) -> Path | None:
    """Draw one sparse waveform panel per unit and save the figure."""
    if units.empty:
        return None
    nrows = int(np.ceil(len(units) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.5 * nrows), squeeze=False)
    used_axes = axes.flat[: len(units)]
    sw.plot_unit_waveforms(
        analyzer,
        unit_ids=units["unit_id"].tolist(),
        axes=list(used_axes),
        max_spikes_per_unit=SPIKES_PER_UNIT,
        backend="matplotlib",
    )
    # Replace the default titles with the numbers that matter for the rule.
    for ax, (_, unit) in zip(used_axes, units.iterrows()):
        ax.set_title(
            f"unit {unit.unit_id}\nSNR {unit.snr:.1f} | ISI {unit.isi_violations_ratio:.2f}\n"
            f"{unit.num_spikes:,} spikes | presence {unit.presence_ratio:.2f}",
            fontsize=8,
        )
    for ax in axes.flat[len(units) :]:
        ax.set_visible(False)
    fig.suptitle(title)
    # Leave room at the top so the figure title doesn't cover the panel titles.
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def pick_rejected_examples(units: pd.DataFrame, rng) -> pd.DataFrame:
    """One rejected unit per criterion that failed, so each cutoff is shown."""
    examples = []
    for criterion in ["pass_isi", "pass_spikes", "pass_snr"]:
        failed_only_this = units[~units[criterion] & (units[["pass_isi", "pass_spikes", "pass_snr"]].sum(axis=1) == 2)]
        failed_this = failed_only_this if not failed_only_this.empty else units[~units[criterion]]
        if not failed_this.empty:
            examples.append(failed_this.sample(n=min(2, len(failed_this)), random_state=rng))
    return pd.concat(examples) if examples else units.iloc[0:0]


def duplicate_pairs(units: pd.DataFrame, all_units: pd.DataFrame, rng) -> pd.DataFrame:
    """Good units flagged as possible duplicates, each followed by its partner."""
    flagged = units[units["possible_duplicate_of"].notna()]
    flagged = flagged.sample(n=min(NUM_DUPLICATE_PAIRS, len(flagged)), random_state=rng)
    rows = []
    for _, unit in flagged.iterrows():
        partner = all_units[all_units["unit_id"] == int(unit.possible_duplicate_of)]
        rows += [unit.to_frame().T, partner]
    return pd.concat(rows) if rows else units.iloc[0:0]


def main():
    table_path = OUTPUT_DIR / f"good_cells_{SESSION}.csv"
    if not table_path.exists():
        raise SystemExit(f"{table_path} not found. Run good_cells.py first.")
    table = pd.read_csv(table_path)

    for probe, areas in AREAS_TO_PLOT.items():
        print(f"Loading {probe} ...")
        analyzer = load_analyzer(SESSION, probe)
        probe_units = table[table["probe"] == probe]

        for area in areas:
            rng = np.random.default_rng(RANDOM_SEED)
            in_area = probe_units[probe_units["area"] == area]
            good = in_area[in_area["good"]]
            rejected = in_area[~in_area["good"]]
            figures = {
                "good": (
                    good.sample(n=min(NUM_GOOD_EXAMPLES, len(good)), random_state=rng),
                    f"{area}: random good units ({len(good)} of {len(in_area)} are good)",
                ),
                "rejected": (
                    pick_rejected_examples(rejected, rng),
                    f"{area}: rejected units, two per failed criterion",
                ),
                "low_presence": (
                    good[good["flag_low_presence"]].head(NUM_FLAGGED_EXAMPLES),
                    f"{area}: good units flagged for low presence (kept)",
                ),
                "duplicates": (
                    duplicate_pairs(good, probe_units, rng),
                    f"{area}: possible duplicates, each next to its partner",
                ),
            }
            for name, (units, title) in figures.items():
                path = OUTPUT_DIR / f"task1_waveforms_{SESSION}_{area}_{name}.png"
                ncols = 2 if name == "duplicates" else 4
                saved = plot_units(analyzer, units, title, path, ncols=ncols)
                print(f"  {area} {name}: {'saved ' + str(saved) if saved else 'nothing to plot'}")


if __name__ == "__main__":
    main()
