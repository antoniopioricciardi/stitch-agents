# Closed-loop rollouts on Super Mario Bros with the same preprocessing as training.
# `policy` is any function x (N, 3, size, size) uint8 on DEVICE -> logits (N, n_classes), so native and
# stitched agents go through the same loop. Actions are sampled from the softmax (the reference
# default), and each is held for SKIP NES frames.
# Speed: rollouts are CPU-bound small ops; run with OMP_NUM_THREADS=1 (faster than the default, and
# several rollout scripts can then run in parallel without oversubscribing the cores).
import numpy as np
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from data import SIZE, SKIP, merge, resize

MAX_DECISIONS = 3000  # the in-game timer (400) ends an episode after ~2400 decisions anyway


@torch.no_grad()
def rollout(policy, rom_version, n_episodes, seed, device, level="1-1", size=SIZE):
    # returns per-episode max x_pos (n_episodes,) and flag_get (n_episodes,)
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v{rom_version}"), COMPLEX_MOVEMENT)
    gen = torch.Generator(device=device).manual_seed(seed)
    max_x, flags = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        img = resize(obs[None], size).to(device)  # (1, 3, size, size)
        prev = torch.zeros_like(img)  # zeros before the start, as in training
        best, flag = 40, False
        for _ in range(MAX_DECISIONS):
            logits = policy(merge(img, prev))
            a = int(torch.multinomial(torch.softmax(logits, 1), 1, generator=gen))
            for _ in range(SKIP):
                obs, _, done, info = env.step(a)  # class a == COMPLEX_MOVEMENT index a
                if done:
                    break
            best, flag = max(best, info["x_pos"]), info["flag_get"]
            if done:
                break
            prev, img = img, resize(obs[None], size).to(device)
        max_x.append(best)
        flags.append(flag)
    env.close()
    return np.array(max_x), np.array(flags)
