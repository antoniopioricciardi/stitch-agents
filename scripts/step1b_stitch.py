"""Step 1b: stitch DP agents across two visual domains (cam0 <-> a second domain) and measure collapse.

Usage: uv run python scripts/step1b_stitch.py <row-name> <domain> <cam0_run_dir> <domain_run_dir> <labels.pt> [--offline-only]
       [--domain0=cam0] [--only-dir=10] [--src-demos=50] [--maps=identity,saps,action_pairs]   (also: affine, mlp, k16_half, nn_orth, nn_affine, nn_affine_rescale, nn_mnn_affine;
       the map-class check uses
       --maps=saps,affine,mlp)
  domain: the second domain (see ENV / H5)
  run dirs: results/<run>/ with runs/<name>/checkpoints/<final>.pt (largest numeric tag = final weights)
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

For each direction u -> v (cam0 encoder -> second-domain controller, and back), stitched agent = C_v(T(E_u(o)), state):
agent_v with its visual_encoder replaced by [E_u, z -> z @ R.T + b], T applied per frame.
  Aligners, fitted on z of the current frames of demos 0-99 (the training demos) of both domains:
    identity; SAPS (paired: same demo and step in both domains, i.e. the same state); action_pairs (labels = the
    action-chunk clusters, <=100 pairs per cluster, N_DRAWS draws; closed loop uses draw 0).
    Map-class check, also on the paired frames: affine (least squares, no orthogonality) and mlp (z -> affine(z) +
    MLP(z), 256 -> 512 -> 256 ReLU; the affine part starts at the least-squares fit, so the class contains affine;
    MSE, Adam, early-stopped on the pairs of 10 held-out fit demos).
    k16_half: action_pairs (K = 16, <=100 pairs per cluster, N_DRAWS draws, orthogonal) with source frames from demos
    0-49 and target frames from demos 50-99, like every label fit from Step 1b round 1 on.
    nn_orth / nn_affine (label-based, scripts/step1b_label_maps.py): source frames of demos 0-49, each paired with the
    target frame of demos 50-99 whose z-scored 8-step chunk is nearest; orthogonal / affine least squares.
    nn_affine_rescale: nn_affine, then each output dimension given the target's per-dimension mean and std (stats of
    all source frames of demos 0-49 mapped vs all target frames of demos 50-99); nn_mnn_affine: affine on the
    mutual-nearest-neighbour pairs only (scripts/step1b_label_maps2.py).
    nn_affine_pca16: nn_affine in the top-16 PCA subspace of z (z's effective rank is 3-4, so the map stays well-posed
    with few demos): PCA on each side's fit frames (source Zs[src], target Zt[tgt]), a 16 -> 16 affine map on the
    projected nearest-chunk pairs, lifted back with the target basis (outside the subspace: the target mean).
    *_pre (k16_half_pre, nn_orth_pre, nn_affine_pre): the same label maps with source and target frames restricted to
    the frames before the grasp closes (the first gripper-close action of each demo; reach and grasp only).
  --domain0: the controller-side domain (default cam0; e.g. xarm_cam0, goal_cam0). --only-dir=10: only the domain-1
    encoder -> domain-0 controller direction, z residual and closed loop only (no offline chunk metrics: the domain-1
    run's own agent cannot play domain 1's robot or task there). --src-demos=N: label fits use source demos 0..N-1
    only (data-efficiency curve; target side unchanged, demos 50-99). --save-maps: also writes each direction's affine maps (R, b; first
    draw) to OUT/maps_<u>_to_<v>.npz as <map>_R, <map>_b.
  --episodes=N: closed-loop episodes per map (default 250).
  Held-out z-space residual per map: ||T(z_s) - z_t||^2 / ||z_t - mean||^2 on the paired held-out frames.
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
from stitch.align import action_pairs, fit_action_pairs, fit_affine_paired, fit_identity, fit_procrustes_paired
from stitch.labels import assign, frame_chunks, standardise
from stitch.models import swap_dino

ENV = {"cam0": "stitch.envs:StitchPickCubeLollipopNoGrasp-v1", "cam1": "stitch.envs:StitchPickCubeLollipopNoGraspCam1-v1",
       "cam2": "stitch.envs:StitchPickCubeLollipopNoGraspCam2-v1", "look1": "stitch.envs:StitchPickCubeLollipopNoGraspLook1-v1",
       "look2": "stitch.envs:StitchPickCubeLollipopNoGraspLook2-v1", "look2light1": "stitch.envs:StitchPickCubeLollipopNoGraspLook2Light1-v1",
       "cam3": "stitch.envs:StitchPickCubeLollipopNoGraspCam3-v1",
       "xarm_cam0": "stitch.envs:StitchPickCubeLollipopNoGraspXarm-v1", "xarm_look2": "stitch.envs:StitchPickCubeLollipopNoGraspLook2Xarm-v1",
       "goal_cam0": "stitch.envs:StitchPickCubeLollipopNoGraspGoal-v1", "goal_look2": "stitch.envs:StitchPickCubeLollipopNoGraspLook2Goal-v1"}
TRAJ = "trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
H5 = {"cam0": f"/home/ricc/projects/labelstitch-step1/results/20260930_dp_ours_demos_default_panda_cam0_lollipop/{TRAJ}",
      "cam1": f"results/20261001_dp_ours_demos_default_panda_cam1_lollipop/{TRAJ}",
      "cam2": f"results/20261001_dp_ours_demos_default_panda_cam2_lollipop/{TRAJ}",
      "look1": f"results/20261001_dp_ours_demos_default_panda_cam0_look1_lollipop/{TRAJ}",
      "look2": f"results/20261002_dp_ours_demos_default_panda_cam0_look2_lollipop/{TRAJ}",
      "look2light1": f"results/20261002_dp_ours_demos_default_panda_cam0_look2_light1_lollipop/{TRAJ}",
      "cam3": f"results/20261002_dp_ours_demos_default_panda_cam3_lollipop/{TRAJ}",
      "xarm_cam0": f"results/20261003_dp_ours_demos_default_xarm6_cam0_lollipop/{TRAJ}",
      "xarm_look2": f"results/20261003_dp_ours_demos_default_xarm6_cam0_look2_lollipop/{TRAJ}",
      "goal_cam0": f"results/20261003_dp_ours_demos_goal_panda_cam0_lollipop/{TRAJ}",
      "goal_look2": f"results/20261003_dp_ours_demos_goal_panda_cam0_look2_lollipop/{TRAJ}"}
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
    # tcp (N, 7) for the pairing check, demo (N,) demo index; one row per current frame t = 0..L-2 of each demo, obs frames [t-1, t]
    # (t-1 clamped to 0, as the baseline pads), the same frames and chunks as in training (stitch/labels.py)
    process = partial(convert_obs, concat_fn=partial(np.concatenate, axis=-1),
                      transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
                      state_obs_extractor=build_state_obs_extractor(ENV[D[cam]]), depth=False)
    rgb, idxs, state, chunks, tcp, demo, n = [], [], [], [], [], [], 0
    with h5py.File(H5[D[cam]], "r") as f:
        for i in demos:
            if f"traj_{i}" not in f:  # the goal-variant demo set has 496 demos (traj_0-495)
                continue
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
            demo.append(np.full(len(t), i))
    return (torch.from_numpy(np.concatenate(rgb)), np.concatenate(idxs), torch.from_numpy(np.concatenate(state)).float(),
            np.concatenate(chunks), np.concatenate(tcp), np.concatenate(demo))


def load_agent(run_dir):
    # the Agent only needs the spaces (no eval-env workers): state (2, S), rgb (2, 128, 128, 3), action in [-1, 1]^4
    ckpt = max(glob.glob(str(Path(run_dir) / "runs/*/checkpoints/[0-9]*.pt")), key=lambda p: int(Path(p).stem))
    sd = torch.load(ckpt, map_location=DEV)["ema_agent"]
    # state size S from the U-Net's FiLM input: cond = diffusion step embedding (64) + obs_horizon * (256 + S)
    cond = sd["noise_pred_net.down_modules.0.0.cond_encoder.1.weight"].shape[1]
    S = (cond - 64) // 2 - 256
    spaces_s = type("E", (), dict(
        single_observation_space=spaces.Dict(state=spaces.Box(-np.inf, np.inf, (2, S)), rgb=spaces.Box(0, 255, (2, 128, 128, 3), np.uint8)),
        single_action_space=spaces.Box(-1, 1, (4,))))()
    agent = swap_dino(train_rgbd.Agent(spaces_s, train_rgbd.Args()).to(DEV), sd)  # Step 3 agents have a DINO encoder
    agent.load_state_dict(sd)
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


def pca_affine(Zs_all, Zt_all, Ps, Pt, k=16):
    # affine map in the top-k PCA subspaces -> (R, b) on the full z: z -> mu_t + Ut (A Us^T (z - mu_s) + c)
    # Zs_all / Zt_all: each side's fit frames (PCA), Ps / Pt: the paired latents (rows correspond)
    mu_s, mu_t = Zs_all.mean(0), Zt_all.mean(0)
    Us = np.linalg.svd(Zs_all - mu_s, full_matrices=False)[2][:k].T  # (d, k)
    Ut = np.linalg.svd(Zt_all - mu_t, full_matrices=False)[2][:k].T
    A, c = fit_affine_paired((Ps - mu_s) @ Us, (Pt - mu_t) @ Ut)  # k -> k
    R = Ut @ A @ Us.T
    return R, mu_t + Ut @ c - R @ mu_s


def rescale(R, b, Zs_all, Zt_all):
    # (R, b) -> (R', b'): mapped z standardised per dimension, then given the target's per-dimension mean and std
    # (copied from scripts/step1b_label_maps2.py)
    M = Zs_all @ R.T + b
    s = Zt_all.std(0) / (M.std(0) + 1e-8)
    return R * s[:, None], (b - M.mean(0)) * s + Zt_all.mean(0)


class MLPMap(nn.Module):
    # z -> affine(z) + MLP(z): the affine part starts at the least-squares fit, the MLP's last layer at zero
    def __init__(self, R, b, hidden=512):
        super().__init__()
        self.lin = affine(R, b)
        self.mlp = nn.Sequential(nn.Linear(R.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, R.shape[0])).to(DEV)
        nn.init.zeros_(self.mlp[2].weight), nn.init.zeros_(self.mlp[2].bias)

    def forward(self, z):
        return self.lin(z) + self.mlp(z)


def fit_mlp_paired(Zs, Zt, demo, seed=0, n_val=10, patience=20, max_epochs=500):
    # paired latents Zs, Zt (N, d), demo (N,) -> MLPMap with the best validation MSE (val = pairs of n_val random demos)
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    val = np.isin(demo, g.choice(np.unique(demo), n_val, replace=False))
    T = MLPMap(*fit_affine_paired(Zs[~val], Zt[~val]))
    xs, xt = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in (Zs, Zt))
    tr, va = torch.from_numpy(~val).to(DEV), torch.from_numpy(val).to(DEV)
    opt = torch.optim.Adam(T.parameters(), lr=1e-3)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(max_epochs):
        for i in torch.randperm(int(tr.sum()), device=DEV).split(256):
            loss = ((T(xs[tr][i]) - xt[tr][i]) ** 2).sum(1).mean()
            opt.zero_grad(), loss.backward(), opt.step()
        with torch.no_grad():
            v = ((T(xs[va]) - xt[va]) ** 2).sum(1).mean().item()
        if v < best:
            best, best_state, bad = v, copy.deepcopy(T.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    T.load_state_dict(best_state)
    return T.eval(), dict(val_mse=best, epochs=epoch + 1)


# the guard is needed: the eval envs are forkserver workers, which re-import this script
if __name__ == "__main__":
    ROW, DOMAIN, LABELS = sys.argv[1], sys.argv[2], sys.argv[5]
    opt = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    D = {0: opt("domain0", "cam0"), 1: DOMAIN}  # domain index -> name
    ONLY_DIR = opt("only-dir", "") == "10"
    SRC_DEMOS = int(opt("src-demos", "50"))
    N_EPISODES = int(opt("episodes", N_EPISODES))  # closed-loop episodes per map (default 250)
    RUNS = {0: sys.argv[3], 1: sys.argv[4]}
    OFFLINE_ONLY = "--offline-only" in sys.argv
    MAPS = next((a.split("=")[1] for a in sys.argv if a.startswith("--maps=")), "identity,saps,action_pairs").split(",")
    OUT = Path("results") / f"{date.today():%Y%m%d}_step1b_stitch_{ROW}_{DOMAIN}"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lab = torch.load(LABELS)
    label = lambda ch: assign(torch.from_numpy(standardise(ch, lab["mean"].numpy(), lab["std"].numpy())).float(),
                              lab["centroids"].float()).numpy()

    tmp = gym.make(ENV[D[0]], **ENV_KWARGS)  # "module:EnvId" imports stitch.envs
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
    fit_demo = fit[0][5]
    # continuous action pairs (nn_*): source frames from demos 0-49, nearest z-scored chunk among target demos 50-99
    X = torch.from_numpy(standardise(fit[0][3], lab["mean"].numpy(), lab["std"].numpy())).float()
    src, tgt = np.where(fit_demo < SRC_DEMOS)[0], np.where(fit_demo >= 50)[0]
    dX = torch.cdist(X[src], X[tgt])
    nn_tgt = tgt[dX.argmin(1).numpy()]
    mnn = (dX.argmin(0)[dX.argmin(1)] == torch.arange(len(src))).numpy()  # source i is the nearest of its own match
    # pre-grasp frames: before the first gripper-close action (raw action[-1] < 0) of each demo
    grip, pre = fit[0][3][:, 0, -1], np.zeros(len(fit_demo), bool)
    for dm in np.unique(fit_demo):
        idx = np.where(fit_demo == dm)[0]  # this demo's frames, in time order
        pre[idx] = np.cumsum(grip[idx] < 0) == 0
    src_pre, tgt_pre = src[pre[src]], tgt[pre[tgt]]
    nn_tgt_pre = tgt_pre[torch.cdist(X[src_pre], X[tgt_pre]).argmin(1).numpy()]
    del fit  # frees the fit frames (~0.4 GB per domain)
    m = dict(checkpoints={D[c]: ckpts[c] for c in (0, 1)}, label_sizes=np.bincount(y_fit, minlength=16).tolist(),
             geometry={D[c]: geometry(Z[(c, c)], y_fit) for c in (0, 1)}, stitch={})
    print(json.dumps(m["geometry"], indent=1), flush=True)

    for u, v in (((1, 0),) if ONLY_DIR else ((0, 1), (1, 0))):
        Zs, Zt = Z[(u, u)], Z[(v, v)]
        fits = {"identity": lambda: [affine(*fit_identity(Zs, Zt))], "saps": lambda: [affine(*fit_procrustes_paired(Zs, Zt))],
                "affine": lambda: [affine(*fit_affine_paired(Zs, Zt))],
                "action_pairs": lambda: [affine(*fit_action_pairs(Zs, Zt, y_fit, y_fit, np.random.default_rng(s))) for s in range(N_DRAWS)],
                "mlp": lambda: [fit_mlp_paired(Zs, Zt, fit_demo)],
                "k16_half": lambda: [affine(*fit_procrustes_paired(Zs[src[ia]], Zt[tgt[ib]])) for ia, ib in
                                     (action_pairs(y_fit[src], y_fit[tgt], np.random.default_rng(s)) for s in range(N_DRAWS))],
                "nn_orth": lambda: [affine(*fit_procrustes_paired(Zs[src], Zt[nn_tgt]))],
                "nn_affine": lambda: [affine(*fit_affine_paired(Zs[src], Zt[nn_tgt]))],
                "nn_affine_pca16": lambda: [affine(*pca_affine(Zs[src], Zt[tgt], Zs[src], Zt[nn_tgt]))],
                "nn_affine_rescale": lambda: [affine(*rescale(*fit_affine_paired(Zs[src], Zt[nn_tgt]), Zs[src], Zt[tgt]))],
                "nn_mnn_affine": lambda: [affine(*fit_affine_paired(Zs[src[mnn]], Zt[nn_tgt[mnn]]))],
                "k16_half_pre": lambda: [affine(*fit_procrustes_paired(Zs[src_pre[ia]], Zt[tgt_pre[ib]])) for ia, ib in
                                         (action_pairs(y_fit[src_pre], y_fit[tgt_pre], np.random.default_rng(s)) for s in range(N_DRAWS))],
                "nn_orth_pre": lambda: [affine(*fit_procrustes_paired(Zs[src_pre], Zt[nn_tgt_pre]))],
                "nn_affine_pre": lambda: [affine(*fit_affine_paired(Zs[src_pre], Zt[nn_tgt_pre]))]}
        maps, map_info = {}, {}
        for name in MAPS:
            maps[name] = fits[name]()
            if name == "mlp":
                maps[name], map_info[name] = [maps[name][0][0]], maps[name][0][1]
        rgb, state = off[u][0][off[u][1][sel]], off[u][2][sel]  # (N, 2, 3, H, W), (N, 2, S)
        zu = encode(agents[u].visual_encoder, rgb.flatten(0, 1)).reshape(len(sel), 2, -1)  # (N, 2, 256)
        zv_cur = encode(agents[v].visual_encoder, off[v][0][off[v][1][sel][:, -1]])  # (N, 256): paired target z
        res = dict(aligners={})
        if ONLY_DIR:  # z residual only
            for name, Ts in maps.items():
                with torch.no_grad():
                    resid = [(((T(zu[:, -1]) - zv_cur) ** 2).sum() / ((zv_cur - zv_cur.mean(0)) ** 2).sum()).item() for T in Ts]
                res["aligners"][name] = dict(z_residual=float(np.mean(resid)), z_residual_draws=resid)
        native = None if ONLY_DIR else denoise(agents[u], zu, state, seed=1)
        expert = torch.from_numpy(off[u][3][sel]).to(DEV)
        per_frame = lambda a, b: (a - b).flatten(1).norm(dim=1)  # (N,)
        # vision sensitivity of each frame: how much the native chunk moves when z comes from a random other frame
        # (state kept). Top 30% = the frames where the encoder drives the action, i.e. where stitching matters.
        for name, Ts in ({} if ONLY_DIR else maps).items():
            if not res.get("chance_z"):
                sens = per_frame(native, denoise(agents[u], zu[perm], state, seed=1))
                top = sens >= sens.quantile(1 - TOP_SENSITIVE)
                dist = lambda a, b: [per_frame(a, b).mean().item(), per_frame(a, b)[top].mean().item()]  # [all, top 30%]
                res.update(native_vs_expert=dist(native, expert), chance_frame=dist(native, native[perm]),
                           chance_z=[sens.mean().item(), sens[top].mean().item()],
                           noise_floor=dist(native, denoise(agents[u], zu, state, seed=2)))
            with torch.no_grad():
                ds = [dist(denoise(agents[v], T(zu), state, seed=1), native) for T in Ts]
                resid = [(((T(zu[:, -1]) - zv_cur) ** 2).sum() / ((zv_cur - zv_cur.mean(0)) ** 2).sum()).item() for T in Ts]
            res["aligners"][name] = dict(offline_dist=np.mean(ds, 0).tolist(), offline_dist_draws=ds,
                                         z_residual=float(np.mean(resid)), **map_info.get(name, {}))
        print(f"{D[u]} enc -> {D[v]} ctrl", json.dumps(res, indent=1), flush=True)

        if not OFFLINE_ONLY:
            if u not in envs:
                envs[u] = make_eval_envs(ENV[D[u]], 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                                         wrappers=[FlattenRGBDObservationWrapper])
            for name, Ts in maps.items():
                stitched = copy.deepcopy(agents[v])
                stitched.visual_encoder = nn.Sequential(copy.deepcopy(agents[u].visual_encoder), Ts[0])
                ev = evaluate(N_EPISODES, stitched, envs[u], DEV, "physx_cpu", progress_bar=False)
                res["aligners"][name].update(success_once=float(ev["success_once"].mean()),
                                             success_at_end=float(ev["success_at_end"].mean()), episodes=len(ev["success_once"]))
                print(f"{D[u]} enc -> {D[v]} ctrl, {name}:", res["aligners"][name], flush=True)
                del stitched
            envs.pop(u).close()  # 10 workers per domain: two sets at once exceed 16 GB
        if "--save-maps" in sys.argv:  # (R, b) of each affine map (first draw), for fine-tuning from a stitched start
            np.savez(OUT / f"maps_{D[u]}_to_{D[v]}.npz", **{f"{n}_{k}": t for n, Ts in maps.items() if isinstance(Ts[0], nn.Linear)
                                                         for k, t in (("R", Ts[0].weight.detach().cpu().numpy()), ("b", Ts[0].bias.detach().cpu().numpy()))})
        m["stitch"][f"{D[u]}_enc_to_{D[v]}_ctrl"] = res
        json.dump(m, open(OUT / "metrics.json", "w"), indent=1)

    for e in envs.values():
        e.close()
    json.dump(dict(row=ROW, domain=DOMAIN, runs={D[c]: RUNS[c] for c in (0, 1)}, labels=LABELS,
                   env={D[c]: ENV[D[c]] for c in (0, 1)}, h5={D[c]: H5[D[c]] for c in (0, 1)},
                   fit_demos=[FIT_DEMOS.start, FIT_DEMOS.stop], heldout_demos=[HELDOUT_DEMOS.start, HELDOUT_DEMOS.stop],
                   maps=MAPS, only_dir=ONLY_DIR, src_demos=SRC_DEMOS, n_pre_src=len(src_pre), n_pre_tgt=len(tgt_pre), n_off=N_OFF, top_sensitive=TOP_SENSITIVE, n_draws=N_DRAWS, n_episodes=N_EPISODES, seed=SEED, closed_loop=not OFFLINE_ONLY),
              open(OUT / "config.json", "w"), indent=1)
