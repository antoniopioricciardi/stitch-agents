# Replay the recorded v0 demos in other ROM versions of the same level.
#   v0: must reproduce the recording exactly (same frames, same x_pos) -> checks the demos are valid.
#   v1 ("downsample": no background) and v2 ("pixel": blocky graphics) have the same dynamics as v0,
#   so the replay gives the same trajectory rendered differently: exactly paired frames across domains.
# Output: data/demos_v{1,2}/<same file name>.npz, same keys as the recording, obs re-rendered.
from pathlib import Path

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

DATA = Path(__file__).parent / "data"
VERSIONS = [0, 1, 2]


def replay(level, version, actions, skip):
    # returns obs (T, 240, 256, 3) seen before each decision, x_pos (T,), and whether the env
    # terminated exactly on the last action (as it did during recording)
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v{version}"), COMPLEX_MOVEMENT)
    obs = env.reset()
    x = 40  # the recorder also starts from 40 (x before the first step)
    obs_buf, x_buf = [], []
    done = False
    for t, a in enumerate(actions):
        assert not done, f"v{version}: episode ended early at step {t}/{len(actions)}"
        obs_buf.append(obs.copy())
        x_buf.append(x)
        for _ in range(skip):
            obs, _, done, info = env.step(int(a))
            x = info["x_pos"]
            if done:
                break
    env.close()
    return np.stack(obs_buf), np.array(x_buf), done


for f in sorted((DATA / "demos").glob("*.npz")):
    d = np.load(f)
    level, skip = str(d["level"]), int(d["skip"])
    for v in VERSIONS:
        obs, x_pos, done = replay(level, v, d["action"], skip)
        x_ok = np.array_equal(x_pos, d["x_pos"])
        if v == 0:
            print(f"{f.name}: v0 x_pos match {x_ok}, frames match {np.array_equal(obs, d['obs'])}, "
                  f"ends on last action {done}")
            continue
        print(f"{'':{len(f.name)}}  v{v} x_pos match {x_ok}, ends on last action {done}")
        out = DATA / f"demos_v{v}"
        out.mkdir(exist_ok=True)
        np.savez_compressed(out / f.name, **{k: d[k] for k in d.files if k != "obs"}, obs=obs, rom_version=v)
