# Record human demos of Super Mario Bros 1-1 with the keyboard.
# One attempt (death or flag) = one episode = one .npz in data/demos/.
#
# Controls: arrows to move (DOWN / UP too), A = jump, S = run. Q or ESC quits (the unfinished
# episode is discarded).
#
# The key state is read once per decision and held for SKIP NES frames; one sample per decision.
# SKIP = 1 records every frame (60 Hz), which is much easier to play: the policy's frame skip is
# applied later, when building the training set (take every k-th frame and its action). Actions are
# the 12 COMPLEX_MOVEMENT ones; COMPLEX_MOVEMENT[:7] == SIMPLE_MOVEMENT, so the action set can also
# be coarsened at training time.
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "1-1"  # e.g. uv run ... record_demos.py 1-2
SKIP = 1  # record every frame; the policy's skip (e.g. 4) is applied at training time
SPEED = 1.0  # 1.0 = real NES speed (60 frames/s); lower it (e.g. 0.5) to make playing easier
SCALE = 3  # window scale factor
OUT_DIR = Path(__file__).parent / "data" / "demos"


def keys_to_action(keys):
    # COMPLEX_MOVEMENT: 0 NOOP, 1 right, 2 right+A, 3 right+B, 4 right+A+B, 5 A, 6 left,
    # 7 left+A, 8 left+B, 9 left+A+B, 10 down, 11 up
    jump, run = keys[pygame.K_a], keys[pygame.K_s]
    if keys[pygame.K_RIGHT]:
        return 4 if jump and run else 3 if run else 2 if jump else 1
    if keys[pygame.K_LEFT]:
        return 9 if jump and run else 8 if run else 7 if jump else 6
    if jump:
        return 5
    if keys[pygame.K_DOWN]:
        return 10
    if keys[pygame.K_UP]:
        return 11
    return 0


def show(screen, frame):
    # frame: (240, 256, 3) uint8; pygame surfaces are (W, H)
    surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
    screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
    pygame.display.flip()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{LEVEL}-v0"), COMPLEX_MOVEMENT)

    pygame.init()
    screen = pygame.display.set_mode((256 * SCALE, 240 * SCALE))
    clock = pygame.time.Clock()

    # next number after the highest existing one (per level), so deleting or renaming files can
    # never make a new episode reuse, or overwrite, an existing number
    nums = [int(f.name.split("_ep")[1][:3]) for f in OUT_DIR.glob(f"{LEVEL}_ep*.npz")]
    episode = max(nums, default=-1) + 1
    quit_ = False
    while not quit_:
        # No seed: nes-py's reset takes none and SMB 1-1 is deterministic given the inputs.
        obs = env.reset()
        info = {"x_pos": 40, "flag_get": False}
        pygame.display.set_caption(f"{LEVEL} | episode {episode} | speed {SPEED}")
        obs_buf, act_buf, x_buf = [], [], []
        done = False
        while not done:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
                quit_ = True
                break
            a = keys_to_action(keys)
            obs_buf.append(obs.copy())  # copy: nes-py reuses the screen buffer
            act_buf.append(a)
            x_buf.append(info["x_pos"])
            for _ in range(SKIP):
                obs, _, done, info = env.step(a)
                show(screen, obs)
                clock.tick(60 * SPEED)
                if done:
                    break
        if quit_:
            break

        outcome = "win" if info["flag_get"] else "fail"
        path = OUT_DIR / f"{LEVEL}_ep{episode:03d}_{outcome}.npz"
        np.savez_compressed(
            path,
            obs=np.stack(obs_buf),  # (T, 240, 256, 3) uint8, obs seen before each decision
            action=np.array(act_buf, dtype=np.int64),  # (T,) COMPLEX_MOVEMENT index, held SKIP frames
            x_pos=np.array(x_buf, dtype=np.int64),  # (T,) x position at decision time
            flag_get=info["flag_get"],
            episode=episode,
            level=LEVEL,
            skip=SKIP,
        )
        print(f"saved {path.name}: {len(act_buf)} steps, final x {info['x_pos']}, "
              f"actions {np.bincount(act_buf, minlength=12)}")
        episode += 1
        time.sleep(0.5)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
