"""Run every step in order: uv run main.py"""

import runpy

# Each script still works on its own; this just runs them one after another.
for script in ["good_cells.py", "plot_waveforms.py"]:
    print(f"--- Running {script} ---")
    runpy.run_path(script, run_name="__main__")

print("--- Done. Results are in outputs/ ---")
