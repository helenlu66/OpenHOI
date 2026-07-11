import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.path_config import grab_data_root

npz_file = np.load(grab_data_root() / "data.npz")

for key in npz_file:
    print(key, npz_file[key])
