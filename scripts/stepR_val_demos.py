"""Step R: the fixed validation set for early stopping = demos 400-409 of a DP demo file, copied as traj_0..traj_9
into a new .h5 (the shared demo file is only read), so the vendored dataset loads them with num_traj=None.
Usage: uv run python scripts/stepR_val_demos.py <demos.h5> <name>
Writes results/<date>_stepR_val_demos_<name>/trajectory.h5.
"""
import os
import sys
import time

import h5py

VAL_DEMOS = range(400, 410)

src, name = sys.argv[1], sys.argv[2]
out = f"results/{time.strftime('%Y%m%d')}_stepR_val_demos_{name}"
os.makedirs(out, exist_ok=True)
with h5py.File(src, "r") as f, h5py.File(f"{out}/trajectory.h5", "w") as g:
    for i, d in enumerate(VAL_DEMOS):
        f.copy(f[f"traj_{d}"], g, name=f"traj_{i}")
print(f"{out}/trajectory.h5: demos {VAL_DEMOS.start}-{VAL_DEMOS.stop - 1} of {src}")
