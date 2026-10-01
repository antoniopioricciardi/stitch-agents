"""Step 1b: stitch DP agents across two visual domains (cam0 <-> a second domain) and measure collapse.

Usage: uv run python scripts/step1b_stitch.py <row-name> <domain> <cam0_run_dir> <domain_run_dir> <labels.pt> [--offline-only]
  domain: cam1, cam2 or look1 (the second domain; see ENV / H5)
  run dirs: results/<run>/ with runs/<name>/checkpoints/<final>.pt (largest numeric tag = final weights)
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

For each direction u -> v (cam0 encoder -> second-domain controller, and back), stitched agent = C_v(T(E_u(o)), state):
agent_v with its visual_encoder replaced by [E_u, z -> z @ R.T + b], T applied per frame.
  Aligners, fitted on z of the current frames of demos 0-99 (the training demos) of both domains:
    identity; SAPS (paired: same demo and step in both domains, i.e. the same state); action_pairs (labels = the
    action-chunk clusters, <=100 pairs per cluster, N_DRAWS draws; closed loop uses draw 0).
  Offline (held-out demos 400-497, N_OFF frames of domain u): ||stitched chunk - native chunk|| (L2 over the
    8 executed steps x 4 dims), native = agent_u, both denoised from the same DDPM noise. References:
    chance_z = native vs native with z taken from a random other frame (state kept: the chance level for z; on
    demo frames the state alone predicts much of the chunk); chance_frame = native(i) vs native(j) for a random
    other frame j; noise floor = native vs native with different noise; expert = native vs the demo's own chunk.
    Every offline number is [all frames, the 30% most vision-sensitive frames] (sensitivity = the per-frame chance_z).
  Closed loop: the baseline's evaluate() on domain u's env, N_EPISODES episodes, final checkpoints.
Per agent (training frames, own domain, chunk labels): NC1 = tr(S_W)/tr(S_B), effective rank (participation ratio),
share of z's variance in the top-4 / top-16 principal components (regression-collapse check).
"""
import copy
import glob
import json
import sys
from datetime import date
from functools import partial
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import h5py
import numpy as np
import torch
import torch.nn as nn

import train_rgbd
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import build_state_obs_extractor, convert_obs, load_content_from_h5_file
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from stitch.align import fit_action_pairs, fit_identity, fit_procrustes_paired
from stitch.labels import assign, frame_chunks, standardise

ENV = {"cam0": "stitch.envs:StitchPickCubeLollipopNoGrasp-v1", "cam1": "stitch.envs:StitchPickCubeLollipopNoGraspCam1-v1",
       "cam2": "stitch.envs:StitchPickCubeLollipopNoGraspCam2-v1", "look1": "stitch.envs:StitchPickCubeLollipopNoGraspLook1-v1"}
TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
      "cam1": f"results/20261001_dp_ours_demos_default_panda_cam1_lollipop/{TRAJ}",
      "cam2": f"results/20261001_dp_ours_demos_default_panda_cam2_lollipop/{TRAJ}",
      "look1": f"results/20261001_dp_ours_demos_default_panda_cam0_look1_lollipop/{TRAJ}"}
FIT_DEMOS = range(0, 100)
HELDOUT_DEMOS = range(400, 498)
N_OFF = 2000
TOP_SENSITIVE = 0.3  # share of most vision-sensitive held-out frames for the second offline number
N_DRAWS = 5
N_EPISODES = 250
SEED = 0
DEV = "cuda"
ENV_KWARGS = dict(control_mode="pd_ee_delta_pos", reward_mode="sparse", obs_mode="rgb", render_mode="rgb_array",
                  human_render_camera_configs=dict(shader_pack="default"), max_episode_steps=100)  # as in train_rgbd.py


def load_frames(cam, demos, obs_space):
    # -> rgb (F, 3, H, W) uint8 every frame once, idx (N, 2) rows of rgb, state (N, 2, S), chunks (N, 8, A),
    # tcp (N, 7) for the pairing check; one row per current frame t = 0..L-2 of each demo, obs frames [t-1, t]
    # (t-1 clamped to 0, as the baseline pads), the same frames and chunks as in training (stitch/labels.py)
    process = partial(convert_obs, concat_fn=partial(np.concatenate, axis=-1),
                      transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
                      state_obs_extractor=build_state_obs_extractor(ENV[D[cam]]), depth=False)
    rgb, idxs, state, chunks, tcp, n = [], [], [], [], [], 0
    with h5py.File(H5[D[cam]], "r") as f:
        for i in demos:
            traj = load_content_from_h5_file(f[f"traj_{i}"])
            obs = process(train_rgbd.reorder_keys(traj["obs"], obs_space))
            L = len(traj["actions"])
            t = np.arange(L - 1)
            idx = np.stack([np.maximum(t - 1, 0), t], 1)  # (L-1, 2)
            rgb.append(obs["rgb"])
            idxs.append(idx + n)
            n += len(obs["rgb"])
            state.append(obs["state"][idx])
            chunks.append(frame_chunks(traj["actions"]))
            tcp.append(traj["obs"]["extra"]["tcp_pose"][t])
    return (torch.from_numpy(np.concatenate(rgb)), np.concatenate(idxs), torch.from_numpy(np.concatenate(state)).float(),
            np.concatenate(chunks), np.concatenate(tcp))


# Agent only needs the spaces (no eval-env workers): state (2, 28), rgb (2, 128, 128, 3), action in [-1, 1]^4
SPACES = type("E", (), dict(
    single_observation_space=spaces.Dict(state=spaces.Box(-np.inf, np.inf, (2, 28)), rgb=spaces.Box(0, 255, (2, 128, 128, 3), np.uint8)),
    single_action_space=spaces.Box(-1, 1, (4,))))()


def load_agent(run_dir):
    ckpt = max(glob.glob(str(Path(run_dir) / "runs/*/checkpoints/[0-9]*.pt")), key=lambda p: int(Path(p).stem))
    agent = train_rgbd.Agent(SPACES, train_rgbd.Args()).to(DEV)
    agent.load_state_dict(torch.load(ckpt)["ema_agent"])
    return agent.eval(), ckpt


@torch.no_grad()
def encode(encoder, rgb):
    # rgb (N, 3, H, W) uint8 -> z (N, 256)
    return torch.cat([encoder(x.to(DEV).float() / 255.0) for x in rgb.split(512)])


@torch.no_grad()
def denoise(agent, z, state, seed):
    # z (N, 2, 256), state (N, 2, S) -> executed chunks (N, 8, A); a copy of Agent.get_action from the conditioning
    # on, with all DDPM noise drawn from `seed` so two calls with the same seed share their noise
    torch.manual_seed(seed)
    out = []
    for zb, sb in zip(z.split(500), state.split(500)):
        cond = torch.cat([zb, sb.to(DEV)], -1).flatten(1)
        x = torch.randn((len(zb), agent.pred_horizon, agent.act_dim), device=DEV)
        for k in agent.noise_scheduler.timesteps:
            eps = agent.noise_pred_net(sample=x, timestep=k, global_cond=cond)
            x = agent.noise_scheduler.step(model_output=eps, timestep=k, sample=x).prev_sample
        out.append(x[:, agent.obs_horizon - 1:agent.obs_horizon - 1 + agent.act_horizon])
    return torch.cat(out)


def geometry(Z, y):
    # Z (N, d) latents, y (N,) labels -> NC1, effective rank, variance share in the top-4 / top-16 PCs
    classes = np.unique(y)
    M = np.stack([Z[y == c].mean(0) for c in classes])
    Sw = np.mean([((Z[y == c] - M[i]) ** 2).sum(1).mean() for i, c in enumerate(classes)])
    Sb = ((M - M.mean(0)) ** 2).sum(1).mean()
    ev = np.sort(np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T)))[::-1]
    return dict(nc1=float(Sw / Sb), eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()),
                var_top4=float(ev[:4].sum() / ev.sum()), var_top16=float(ev[:16].sum() / ev.sum()))


def affine(R, b):
    lin = nn.Linear(R.shape[1], R.shape[0]).to(DEV)
    lin.weight.data, lin.bias.data = torch.tensor(R, dtype=torch.float32, device=DEV), torch.tensor(b, dtype=torch.float32, device=DEV)
    return lin


# the guard is needed: the eval envs are forkserver workers, which re-import this script
if __name__ == "__main__":
    ROW, DOMAIN, LABELS = sys.argv[1], sys.argv[2], sys.argv[5]
    D = {0: "cam0", 1: DOMAIN}  # domain index -> name
    RUNS = {0: sys.argv[3], 1: sys.argv[4]}
    OFFLINE_ONLY = "--offline-only" in sys.argv
    OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_stitch_{ROW}_{DOMAIN}"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lab = torch.load(LABELS)
    label = lambda ch: assign(torch.from_numpy(standardise(ch, lab["mean"].numpy(), lab["std"].numpy())).float(),
                              lab["centroids"].float()).numpy()

    tmp = gym.make(ENV["cam0"], **ENV_KWARGS)  # "module:EnvId" imports stitch.envs
    obs_space = tmp.observation_space
    tmp.close()
    envs = {}  # eval envs, made only for the closed loop
    agents, ckpts = {}, {}
    for c in (0, 1):
        agents[c], ckpts[c] = load_agent(RUNS[c])

    fit, off = {}, {}
    for c in (0, 1):
        fit[c] = load_frames(c, FIT_DEMOS, obs_space)
        off[c] = load_frames(c, HELDOUT_DEMOS, obs_space)
    for d in (fit, off):  # same demo, same step -> same state and actions in both domains (the SAPS pairing)
        assert np.abs(d[0][4] - d[1][4]).max() < 1e-5 and np.abs(d[0][2].numpy() - d[1][2].numpy()).max() < 1e-5
        assert np.array_equal(d[0][3], d[1][3])
    y_fit, y_off = label(fit[0][3]), label(off[0][3])
    sel = rng.permutation(len(y_off))[:N_OFF]  # held-out frames for the offline metrics
    perm = rng.permutation(N_OFF)  # random other frame, for the chance level

    # z of the current frames: Z[(encoder cam, image cam)], (N, 256)
    Z = {(c, c): encode(agents[c].visual_encoder, fit[c][0][fit[c][1][:, -1]]).cpu().numpy() for c in (0, 1)}
    del fit  # frees the fit frames (~0.4 GB per domain)
    m = dict(checkpoints={D[c]: ckpts[c] for c in (0, 1)}, label_sizes=np.bincount(y_fit, minlength=16).tolist(),
             geometry={D[c]: geometry(Z[(c, c)], y_fit) for c in (0, 1)}, stitch={})
    print(json.dumps(m["geometry"], indent=1), flush=True)

    for u, v in ((0, 1), (1, 0)):
        Zs, Zt = Z[(u, u)], Z[(v, v)]
        maps = {"identity": [fit_identity(Zs, Zt)], "saps": [fit_procrustes_paired(Zs, Zt)],
                "action_pairs": [fit_action_pairs(Zs, Zt, y_fit, y_fit, np.random.default_rng(s)) for s in range(N_DRAWS)]}
        rgb, state = off[u][0][off[u][1][sel]], off[u][2][sel]  # (N, 2, 3, H, W), (N, 2, S)
        zu = encode(agents[u].visual_encoder, rgb.flatten(0, 1)).reshape(len(sel), 2, -1)  # (N, 2, 256)
        native = denoise(agents[u], zu, state, seed=1)
        expert = torch.from_numpy(off[u][3][sel]).to(DEV)
        per_frame = lambda a, b: (a - b).flatten(1).norm(dim=1)  # (N,)
        # vision sensitivity of each frame: how much the native chunk moves when z comes from a random other frame
        # (state kept). Top 30% = the frames where the encoder drives the action, i.e. where stitching matters.
        sens = per_frame(native, denoise(agents[u], zu[perm], state, seed=1))
        top = sens >= sens.quantile(1 - TOP_SENSITIVE)
        dist = lambda a, b: [per_frame(a, b).mean().item(), per_frame(a, b)[top].mean().item()]  # [all, top 30%]
        res = dict(native_vs_expert=dist(native, expert), chance_frame=dist(native, native[perm]),
                   chance_z=[sens.mean().item(), sens[top].mean().item()],
                   noise_floor=dist(native, denoise(agents[u], zu, state, seed=2)), aligners={})
        for name, RBs in maps.items():
            ds = [dist(denoise(agents[v], zu @ torch.tensor(R.T, dtype=torch.float32, device=DEV)
                               + torch.tensor(b, dtype=torch.float32, device=DEV), state, seed=1), native) for R, b in RBs]
            res["aligners"][name] = dict(offline_dist=np.mean(ds, 0).tolist(), offline_dist_draws=ds)
        print(f"{D[u]} enc -> {D[v]} ctrl", json.dumps(res, indent=1), flush=True)

        if not OFFLINE_ONLY:
            if u not in envs:
                envs[u] = make_eval_envs(ENV[D[u]], 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                                         wrappers=[FlattenRGBDObservationWrapper])
            for name, RBs in maps.items():
                stitched = copy.deepcopy(agents[v])
                stitched.visual_encoder = nn.Sequential(copy.deepcopy(agents[u].visual_encoder), affine(*RBs[0]))
                ev = evaluate(N_EPISODES, stitched, envs[u], DEV, "physx_cpu", progress_bar=False)
                res["aligners"][name].update(success_once=float(ev["success_once"].mean()),
                                             success_at_end=float(ev["success_at_end"].mean()), episodes=len(ev["success_once"]))
                print(f"{D[u]} enc -> {D[v]} ctrl, {name}:", res["aligners"][name], flush=True)
                del stitched
            envs.pop(u).close()  # 10 workers per domain: two sets at once exceed 16 GB
        m["stitch"][f"{D[u]}_enc_to_{D[v]}_ctrl"] = res
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)

    for e in envs.values():
        e.close()
    json.dump(dict(row=ROW, domain=DOMAIN, runs={D[c]: RUNS[c] for c in (0, 1)}, labels=LABELS,
                   env={D[c]: ENV[D[c]] for c in (0, 1)}, h5={D[c]: H5[D[c]] for c in (0, 1)},
                   fit_demos=[FIT_DEMOS.start, FIT_DEMOS.stop], heldout_demos=[HELDOUT_DEMOS.start, HELDOUT_DEMOS.stop],
                   n_off=N_OFF, top_sensitive=TOP_SENSITIVE, n_draws=N_DRAWS, n_episodes=N_EPISODES, seed=SEED, closed_loop=not OFFLINE_ONLY),
              open(OUT / "config.json", "w"), indent=1)
