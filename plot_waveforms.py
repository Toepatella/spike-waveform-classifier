"""Task 1: sparse waveform plots of good and rejected units. Run good_cells.py first."""

import matplotlib.pyplot as plt
import pandas as pd
import spikeinterface.widgets as sw

from load_data import load_analyzer

SESSION = "2026-07-21"
AREAS = {"PFC": ["PFC"], "HPC": ["CA3", "DG"]}
UNITS_PER_FIGURE = 8

units = pd.read_csv(f"outputs/good_cells_{SESSION}.csv")

for probe, areas in AREAS.items():
    analyzer = load_analyzer(SESSION, probe)
    for area in areas:
        in_area = units[(units.probe == probe) & (units.area == area)]
        for label, group in [("good", in_area[in_area.good]), ("rejected", in_area[~in_area.good])]:
            # random_state fixes the sample so reruns show the same units
            sample = group.sample(min(UNITS_PER_FIGURE, len(group)), random_state=0)
            fig, axes = plt.subplots(2, 4, figsize=(13, 9))
            sw.plot_unit_waveforms(
                analyzer,
                unit_ids=list(sample.unit_id),
                axes=list(axes.flat[: len(sample)]),
                max_spikes_per_unit=50,  # faint lines; bold line = average
            )
            for ax, unit in zip(axes.flat, sample.itertuples()):
                ax.set_title(
                    f"unit {unit.unit_id}\nSNR {unit.snr:.1f}  ISI {unit.isi_violations_ratio:.2f}"
                    f"  spikes {unit.num_spikes}",
                    fontsize=8,
                )
            fig.suptitle(f"{area}: {label} units")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(f"outputs/waveforms_{area}_{label}.png", dpi=110)
            plt.close(fig)
