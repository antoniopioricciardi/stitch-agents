"""Closed-loop rollout of C(z, proprio) with z = encode(env, obs), or a blind z (zeros / z of a random other frame)."""
import numpy as np
import torch

from stitch.envs import make_env, proprio

EVAL_SEED0 = 10_000  # demos use seeds 0..499; evaluation seeds are disjoint


def pixel_encoder(encoder):
    # encode(env, obs) for a pixel Encoder: z = E(rgb) (1, d)
    device = next(encoder.parameters()).device
    return lambda env, obs: encoder(obs["sensor_data"]["base_camera"]["rgb"].to(device))


@torch.no_grad()
def evaluate(encode, controller, visual, n_episodes=100, z_mode="full", z_bank=None, task="default", robot="panda",
             seed0=EVAL_SEED0, control_mode="pd_ee_delta_pose", hist=1, n_exec=1, obs_mode="rgb"):
    # Runs every episode to the step limit. z_mode: "full" = encode(env, obs); "zeros" = 0; "shuffled" = a random
    # row of z_bank (N, d), the latent of a random training frame, redrawn every step.
    # hist: frames stacked as controller input ([z_{t-1}, z_t], [p_{t-1}, p_t]); the first step repeats itself.
    # The controller predicts a chunk (1, H, a); the first n_exec actions are executed, then it replans.
    # Returns per-episode success at any step, success at the last step, and the first success step (-1 if none).
    device = controller.p_mean.device
    d = (controller.mlp[0].in_features - len(controller.p_mean)) // hist
    env = make_env(visual, task, robot, obs_mode=obs_mode, control_mode=control_mode)
    rng = np.random.default_rng(seed0)
    once, at_end, first = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        zs, ps, queue = [], [], []
        success_t, t, done = -1, 0, False
        while not done:
            if z_mode == "full":
                z = encode(env, obs)
            elif z_mode == "zeros":
                z = torch.zeros(1, d, device=device)
            else:
                z = z_bank[rng.integers(len(z_bank))][None]
            zs, ps = (zs + [z])[-hist:], (ps + [torch.from_numpy(proprio(env))[None].to(device)])[-hist:]
            if not queue:
                zh, ph = [zs[0]] * (hist - len(zs)) + zs, [ps[0]] * (hist - len(ps)) + ps
                queue = list(controller(torch.cat(zh, 1), torch.cat(ph, 1))[0, :n_exec].clamp(-1, 1).cpu().numpy())
            obs, _, _, truncated, info = env.step(queue.pop(0))
            t += 1
            success = bool(info["success"].item())
            if success and success_t < 0:
                success_t = t
            done = bool(truncated)
        once.append(success_t >= 0), at_end.append(success), first.append(success_t)
    env.close()
    return np.array(once), np.array(at_end), np.array(first)
