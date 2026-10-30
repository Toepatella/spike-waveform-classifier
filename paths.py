"""Where the project's data lives.

The recordings are too large for OneDrive and git, so they are kept outside
the project folder. Set the SPIKE_DATA_DIR environment variable to use a
different location (for example on another computer).
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("SPIKE_DATA_DIR", r"C:\Users\tonyt\spike-data"))
