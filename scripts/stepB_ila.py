"""Step B: ILA (Invariance Through Latent Alignment, Yoneda et al., RSS 2022) on cam0 -> look 2, closed loop.

Usage: uv run python scripts/stepB_ila.py <pair> [--steps=10000] [--pretrain=2000] [--episodes=250] [--tag=x]
Run from the repo root with PYTHONPATH=<repo>:<repo>/third_party/maniskill_diffusion_policy.

A port of ila/invr_thru_inf (adapt.py, agent.py, dynamics.py) with the defaults of its config.py: the cam0 agent of
matrix cell (look2, <pair>, bwd) keeps its controller; a copy of its PlainConv encoder E is adapted on unpaired look 2
transitions with
  - WGAN: per step, 5 discriminator updates (RMSprop 1e-6, weights clipped to +-.01) on source latents vs E(target
    frames), then one encoder update maximising D(E(target)) (RMSprop 1e-5);
  - dynamics consistency: inverse head (z_t, z_t+1) -> a_t and forward head (z_t, a_t) -> z_t+1, pretrained on source
    transitions (2000 steps, RMSprop 1e-3, encoder frozen) then frozen; each step one encoder update on the target
    batch with z_t+1 under no_grad, MSE of both heads, two RMSprop optimisers at 1e-6 (as their DynamicsModel).
Heads: their InverseDynamicsHead / ForwardDynamicsHead / Discriminator (trunk Linear + LayerNorm + tanh to 50, MLP 600),
orthogonal init. Deviations (EXPERIMENTS.md): no random-shift augmentation; the heads see (z - mu_src) / s_src
(per-dimension source mean, one global source scale).
Data: source = the original encoder on cam0 demos 0-49 (latent bank + transitions); target = look 2 demos 50-99.
Writes results/<date>_stepB_ila_<pair><tag>/{metrics.json, config.json}: closed loop of the unadapted and adapted agent
in look 2, a training log every 500 steps, held-out z residual vs the original encoder on the same states in cam0.
"""
import copy
import json
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5b_motion_matrix as mm  # sets the domains' demo files in s5.H5
import step5b_stitch as s5
from step1b_stitch import ENV, ENV_KWARGS, encode, load_agent
from stepB_gw_matrix import cell_runs, residual
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

SRC_DEMOS, TGT_DEMOS, HELDOUT_DEMOS = range(0, 50), range(50, 100), range(400, 498)
BATCH, HIDDEN, FEAT = 256, 600, 50
LR = dict(gan_enc=1e-5, gan_disc=1e-6, invm=1e-6, fwdm=1e-6, pretrain=1e-3)
N_DISCR, CLIP = 5, 0.01
DEV = "cuda"


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Discriminator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, FEAT), nn.LayerNorm(FEAT), nn.Tanh())
        self.mlp = nn.Sequential(nn.Linear(FEAT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
        self.apply(weight_init)

    def forward(self, z):
        return self.mlp(self.trunk(z))


class InverseHead(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, a))
        self.apply(weight_init)

    def forward(self, z, z_next):
        return torch.tanh(self.mlp(torch.cat([z, z_next], -1)))


class ForwardHead(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.latent_input = nn.Sequential(nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                          nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.action_input = nn.Sequential(nn.Linear(a, 100), nn.LayerNorm(100), nn.Linear(100, 100), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(HIDDEN + 100, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, d), nn.LayerNorm(d), nn.Tanh())
        self.apply(weight_init)

    def forward(self, z, a):
        return self.mlp(torch.cat([self.latent_input(z), self.action_input(a)], -1))


def transitions(name, demos):
    # -> obs (N, 3, H, W) uint8, act (N, A), next_obs (N, 3, H, W) uint8: every step t = 0..L-1 of each demo
    o, a, o2 = [], [], []
    with h5py.File(s5.H5[name], "r") as f:
        for i in demos:
            g = f[f"traj_{i}"]
            rgb = g["obs/sensor_data/base_camera/rgb"][:].transpose(0, 3, 1, 2)  # (L+1, 3, H, W)
            o.append(rgb[:-1]), a.append(g["actions"][:]), o2.append(rgb[1:])
    return torch.from_numpy(np.concatenate(o)), torch.from_numpy(np.concatenate(a)).float(), torch.from_numpy(np.concatenate(o2))


if __name__ == "__main__":
    PAIR = sys.argv[1]
    opt = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    STEPS, PRETRAIN, N_EPISODES = int(opt("steps", 10000)), int(opt("pretrain", 2000)), int(opt("episodes", 250))
    OUT = Path("results") / f"{date.today():%Y%m%d}_stepB_ila_{PAIR}{opt('tag', '')}"
    OUT.mkdir(parents=True, exist_ok=True)
    SEED = int(PAIR[1:])
    torch.manual_seed(SEED), np.random.seed(SEED)
    run = cell_runs("look2", PAIR)["cam0"]
    agent, ckpt = load_agent(run)
    E0 = agent.visual_encoder  # original encoder, frozen
    E = copy.deepcopy(E0).train()
    px = lambda x: x.to(DEV).float() / 255.0  # uint8 frames -> encoder input, as encode_obs

    so, sa, so2 = transitions("cam0", SRC_DEMOS)
    to, ta, to2 = transitions("look2", TGT_DEMOS)
    with torch.no_grad():
        Zs, Zs2 = encode(E0, so), encode(E0, so2)  # (N, 256) source latents of each transition
    # deviation 2: what the ILA heads see. Per-dimension mean, one global scale (RMS of the per-dimension stds): some
    # units are dead on cam0 frames (std 0), so per-dimension scaling would blow up their look 2 activations.
    mu, sd = Zs.mean(0), Zs.var(0).mean().sqrt()
    nz = lambda z: (z - mu) / sd
    d, A = Zs.shape[1], sa.shape[1]
    sa, ta = sa.to(DEV), ta.to(DEV)
    log = dict(pretrain={}, adapt=[])

    # pretrain the dynamics heads on source transitions (encoder frozen), then freeze them
    inv, fwd = InverseHead(d, A).to(DEV), ForwardHead(d, A).to(DEV)
    opt_h = [torch.optim.RMSprop(inv.parameters(), lr=LR["pretrain"]), torch.optim.RMSprop(fwd.parameters(), lr=LR["pretrain"])]
    for step in range(PRETRAIN):
        i = torch.randint(len(Zs), (BATCH,), device=DEV)
        li, lf = F.mse_loss(inv(nz(Zs[i]), nz(Zs2[i])), sa[i]), F.mse_loss(fwd(nz(Zs[i]), sa[i]), nz(Zs2[i]))
        for o in opt_h:
            o.zero_grad()
        (li + lf).backward()
        for o in opt_h:
            o.step()
    with torch.no_grad():  # pretraining fit: R^2 of each head on all source transitions
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
        log["pretrain"] = dict(inv_r2=r2(torch.cat([inv(nz(Zs[k]), nz(Zs2[k])) for k in torch.arange(len(Zs), device=DEV).split(2048)]), sa),
                               fwd_r2=r2(torch.cat([fwd(nz(Zs[k]), sa[k]) for k in torch.arange(len(Zs), device=DEV).split(2048)]), nz(Zs2)))
    inv.requires_grad_(False), fwd.requires_grad_(False)
    print("pretrain", log["pretrain"], flush=True)

    D = Discriminator(d).to(DEV)
    opt_d = torch.optim.RMSprop(D.parameters(), lr=LR["gan_disc"])
    opt_e = torch.optim.RMSprop(E.parameters(), lr=LR["gan_enc"])
    opt_dyn = [torch.optim.RMSprop(E.parameters(), lr=LR["invm"]), torch.optim.RMSprop(E.parameters(), lr=LR["fwdm"])]
    for step in range(STEPS + 1):
        for _ in range(N_DISCR):  # discriminator: maximise D(source) - D(E(target))
            i, j = torch.randint(len(Zs), (BATCH,), device=DEV), torch.randint(len(to), (BATCH,))
            with torch.no_grad():
                zt = nz(E(px(to[j])))
            loss_d = -D(nz(Zs[i])).mean() + D(zt).mean()
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            for p in D.parameters():
                p.data.clamp_(-CLIP, CLIP)
        # encoder: fool the discriminator, on the last discriminator batch's target frames (as their loop)
        D.requires_grad_(False)
        loss_g = -D(nz(E(px(to[j])))).mean()
        opt_e.zero_grad()
        loss_g.backward()
        opt_e.step()
        D.requires_grad_(True)
        # dynamics consistency on the same target transitions, next latent without gradient
        z = nz(E(px(to[j])))
        with torch.no_grad():
            z2 = nz(E(px(to2[j])))
        li, lf = F.mse_loss(inv(z, z2), ta[j.to(DEV)]), F.mse_loss(fwd(z, ta[j.to(DEV)]), z2)
        for o in opt_dyn:
            o.zero_grad()
        (li + lf).backward()
        for o in opt_dyn:
            o.step()
        if step % 500 == 0:
            log["adapt"].append(dict(step=step, w_est=-float(loss_d), gen=float(loss_g), inv=float(li), fwd=float(lf)))
            print(log["adapt"][-1], flush=True)

    # held-out z residual: the adapted (and the original) encoder on look 2 frames vs the original encoder on the same
    # states in cam0 (paired, evaluation only)
    hu, hv = s5.load("look2", HELDOUT_DEMOS), s5.load("cam0", HELDOUT_DEMOS)
    E.eval()
    with torch.no_grad():
        Zref = encode(E0, hv["rgb"]).cpu().double().numpy()
        I = np.eye(d)
        res = {k: residual(I, np.zeros(d), encode(enc, hu["rgb"]).cpu().double().numpy(), Zref) for k, enc in (("no_adapt", E0), ("ila", E))}
    print("held-out residual", res, flush=True)

    m = dict(checkpoint=ckpt, pretrain=log["pretrain"], heldout_residual=res, log=log["adapt"], closed_loop={})
    if N_EPISODES:
        envs = make_eval_envs(ENV["look2"], 10, "physx_cpu", ENV_KWARGS, dict(obs_horizon=2), video_dir=None,
                              wrappers=[FlattenRGBDObservationWrapper])
        for name, enc in (("no_adapt", E0), ("ila", E)):
            a = copy.deepcopy(agent)
            a.visual_encoder = enc
            ev = evaluate(N_EPISODES, a, envs, DEV, "physx_cpu", progress_bar=False)
            m["closed_loop"][name] = dict(success_once=float(ev["success_once"].mean()), success_at_end=float(ev["success_at_end"].mean()),
                                          episodes=len(ev["success_once"]))
            print(name, m["closed_loop"][name], flush=True)
        envs.close()
    json.dump(m, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(dict(pair=PAIR, seed=SEED, cam0_run=run, steps=STEPS, pretrain_steps=PRETRAIN, batch=BATCH, lr=LR, n_discr=N_DISCR,
                   clip=CLIP, hidden=HIDDEN, feat=FEAT, src_demos=[0, 50], tgt_demos=[50, 100], augment=False,
                   heads_input="(z - mu_src) / s_src, s_src = RMS of per-dim source stds", terms=["wgan", "inverse", "forward"], n_episodes=N_EPISODES),
              open(OUT / "config.json", "w"), indent=1)
