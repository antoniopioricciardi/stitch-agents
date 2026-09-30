"""Step 2e: frames of a goal marker from all cameras at chosen goal positions (no training).

Usage: uv run python scripts/check_goal_marker.py [goal_marker]   (default: lollipop)

Rows: cameras 0-2 x {look0/light0, look2/light1}; columns: goal positions covering the default goal region
(x, y in [-0.1, 0.1], z in [0.02, 0.32]) and the goal variant (y in [0.15, 0.25]). The goal is placed by hand after
a reset with seed 0 (cube and robot as in that episode).
"""
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.envs import CAMERAS, make_env
from mani_skill.utils.structs.pose import Pose

MARKER = sys.argv[1] if len(sys.argv) > 1 else "lollipop"
GOALS = {"default low": [0.0, 0.0, 0.03], "default high": [0.1, -0.1, 0.32], "default mid": [-0.1, 0.1, 0.17],
         "goal variant mid": [0.0, 0.2, 0.17], "goal variant high": [0.1, 0.25, 0.32]}
VISUALS = [(c, look, light) for c in range(len(CAMERAS)) for look, light in [(0, 0), (2, 1)]]
OUT = Path("results") / f"{date.today():%Y%m%d}_step2e_goal_marker_{MARKER}"
OUT.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(len(VISUALS), len(GOALS), figsize=(2.2 * len(GOALS), 2.3 * len(VISUALS)))
for r, v in enumerate(VISUALS):
    env = make_env(v, obs_mode="rgb", control_mode="pd_ee_delta_pos", goal_marker=MARKER)
    env.reset(seed=0)
    for c, (name, p) in enumerate(GOALS.items()):
        env.unwrapped.goal_site.set_pose(Pose.create_from_pq(torch.tensor([p])))
        rgb = env.unwrapped.get_obs()["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
        plt.imsave(OUT / f"cam{v[0]}_look{v[1]}_light{v[2]}_{name.replace(' ', '_')}.png", rgb)
        axes[r, c].imshow(rgb), axes[r, c].axis("off")
        axes[r, c].set_title(f"cam{v[0]} look{v[1]} light{v[2]}\n{name} {p}", fontsize=6)
    env.close()
fig.tight_layout()
fig.savefig(OUT / "grid.png", dpi=150)
print("frames in", OUT)
