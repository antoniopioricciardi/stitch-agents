"""Step 1b: ManiSkill's Diffusion Policy RGB baseline + SupCon on the visual feature z, with action-chunk labels.

The training loop is a copy of the `__main__` block of third_party/maniskill_diffusion_policy/train_rgbd.py
(vendored file untouched; Agent, Args, dataset, env and evaluation code are imported from it). Changes:
  - loss = DP loss + supcon_weight * SupCon(z_t, y_t). z_t: (B, 256) PlainConv output of the current frame (the last
    of the 2 obs frames), captured with a forward hook during the vendored compute_loss. y_t: k-means cluster of the
    executed chunk action_seq[:, 1:9] (stitch/labels.py; centroids from scripts/step1b_labels.py).
    supcon_weight = 0 trains plain DP (the DP loss alone is back-propagated; SupCon is only logged).
  - every log_freq iterations, logs the SupCon value and the gradient norm each term sends into the encoder.
  - checkpoints at total_iters - 10000, total_iters - 5000 and total_iters - 1 (40k / 45k / final for 50k),
    instead of --save_freq.
  - fine-tuning (Step 1b adaptation baselines): --init_ckpt (start from a trained agent), or --init_encoder_ckpt +
    --init_controller_ckpt + --init_map (stitched start: one agent's encoder, an affine map "<maps.npz>:<map name>" from
    step1b_stitch.py --save-maps (z -> z @ R.T + b) or "identity", another agent's controller). --train_part: all | encoder (controller frozen:
    the DP loss trains the encoder through the frozen denoiser) | map (only the affine map) | map_ctrl_last (Step F: the
    map + the denoiser's output side, i.e. its last up block and output conv, 360k parameters). --final_eval_episodes:
    episodes of the evaluation at total_iters (default: num_eval_episodes).
  - early stopping (Step R, secondary number; the headline is the fixed budget total_iters): --val_demo_path (held-out
    demos, from scripts/stepR_val_demos.py) -> every val_freq iterations, on the EMA agent: L2 between the sampled
    executed chunk (fixed seed) and the demo chunk (the selection criterion), and the DP loss with fixed noise and
    timesteps (logged only: it rises while closed-loop success rises, so it is no proxy). The lowest chunk error is
    saved as checkpoints/best_val.pt and evaluated at the end (final_eval_episodes). Writes runs/<name>/best_val.json.
Usage: as run_dp.sh, via scripts/run_dp_supcon.sh.
"""
import copy
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import train_rgbd
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.utils import IterationBasedBatchSampler, build_state_obs_extractor, convert_obs, worker_init_fn
from stitch.labels import CHUNK, assign


@dataclass
class Args(train_rgbd.Args):
    labels: str = ""
    """labels.pt from scripts/step1b_labels.py (k-means centroids, action mean/std)"""
    supcon_weight: float = 0.0
    supcon_tau: float = 0.07
    init_ckpt: str = ""
    init_encoder_ckpt: str = ""
    init_controller_ckpt: str = ""
    init_map: str = ""
    train_part: str = "all"
    final_eval_episodes: int = 0
    val_demo_path: str = ""
    val_freq: int = 500


def supcon(e, y, tau=0.07):
    # copied from mario/models.py: SupCon (SCIL appendix), anchors without positives in the batch contribute 0.
    # e: (N, d) embeddings, y: (N,) labels
    z = F.normalize(e, dim=1)
    logits = z @ z.T / tau
    logits = logits - logits.max(1, keepdim=True).values.detach()  # numerical stability
    not_self = ~torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = (y[:, None] == y[None, :]) & not_self
    log_prob = logits - torch.log((logits.exp() * not_self).sum(1, keepdim=True) + 1e-12)
    n_pos = pos.sum(1).clamp(min=1)
    return -((log_prob * pos).sum(1) / n_pos).mean()


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)).item()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = args.exp_name
    train_rgbd.args = args  # the vendored dataset reads args.control_mode / horizons as a module global

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    train_rgbd.device = device  # Agent.compute_loss reads `device` as a module global

    # chunk labels: centroids (K, CHUNK*A) in z-scored action units
    lab = torch.load(args.labels)
    centroids, a_mean, a_std = lab["centroids"].to(device), lab["mean"].float().to(device), lab["std"].float().to(device)

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=config,
                   name=run_name, save_code=True, group="DiffusionPolicy", tags=["diffusion_policy"])
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth="rgbd" in args.demo_path
    )

    tmp_env = gym.make(args.env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    dataset = train_rgbd.SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device,
        num_traj=args.num_demos
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    agent = train_rgbd.Agent(envs, args).to(device)
    ema_sd = lambda path: torch.load(path, map_location=device)["ema_agent"]
    if args.init_ckpt:  # fine-tune a trained agent
        agent.load_state_dict(ema_sd(args.init_ckpt))
    if args.init_controller_ckpt:  # stitched start: controller (and state handling) of one agent, encoder of another
        agent.load_state_dict(ema_sd(args.init_controller_ckpt))
        agent.visual_encoder.load_state_dict({k[len("visual_encoder."):]: v for k, v in ema_sd(args.init_encoder_ckpt).items()
                                              if k.startswith("visual_encoder.")})
    if args.init_map:  # affine map between encoder and controller, z -> z @ R.T + b
        lin = nn.Linear(256, 256).to(device)
        if args.init_map == "identity":
            R, b = np.eye(256), np.zeros(256)
        else:
            path, name = args.init_map.split(":")  # maps .npz from step1b_stitch.py --save-maps, and a map name
            M = np.load(path)
            R, b = M[f"{name}_R"], M[f"{name}_b"]
        lin.weight.data, lin.bias.data = torch.tensor(R, dtype=torch.float32, device=device), torch.tensor(b, dtype=torch.float32, device=device)
        agent.visual_encoder = nn.Sequential(agent.visual_encoder, lin)
    if args.train_part == "encoder":  # controller frozen
        agent.noise_pred_net.requires_grad_(False)
    if args.train_part == "map":  # only the affine map
        agent.requires_grad_(False)
        agent.visual_encoder[1].requires_grad_(True)
    if args.train_part == "map_ctrl_last":  # Step F (b): the map + the denoiser's output side
        agent.requires_grad_(False)
        agent.visual_encoder[1].requires_grad_(True)
        agent.noise_pred_net.up_modules[-1].requires_grad_(True)
        agent.noise_pred_net.final_conv.requires_grad_(True)

    optimizer = optim.AdamW(params=[p for p in agent.parameters() if p.requires_grad], lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=args.total_iters)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    # same structure as agent (incl. a map); a fresh Agent otherwise, as the vendored loop (keeps its RNG stream)
    ema_agent = copy.deepcopy(agent) if args.init_map else train_rgbd.Agent(envs, args).to(device)

    # z of every encoder call during compute_loss: (B * obs_horizon, 256)
    feats = {}
    agent.visual_encoder.register_forward_hook(lambda m, inp, out: feats.__setitem__("z", out))
    enc_params = [p for p in agent.visual_encoder.parameters() if p.requires_grad]

    # validation set for early stopping: fixed noise and timesteps per sample, so successive losses are comparable
    val_batches = []
    if args.val_demo_path:
        val_set = train_rgbd.SmallDemoDataset_DiffusionPolicy(args.val_demo_path, obs_process_fn, orignal_obs_space,
                                                              include_rgb, include_depth, device, num_traj=None)
        g = torch.Generator(device=device).manual_seed(0)
        for batch in DataLoader(val_set, batch_size=args.batch_size, shuffle=False):
            act = batch["actions"]  # (B, pred_horizon, A)
            noise = torch.randn(act.shape, device=device, generator=g)
            t = torch.randint(0, ema_agent.noise_scheduler.config.num_train_timesteps, (len(act),), device=device, generator=g)
            val_batches.append((batch["observations"], act, noise, t))

    @torch.no_grad()
    def val_loss(model):
        # DP loss: the vendored compute_loss with the stored noise / timesteps, no augmentation.
        # chunk: mean L2 between the sampled executed chunk (B, act_horizon, A) and the demo's, fixed sampling seed
        # (forked RNG, so the training stream is untouched). get_action wants the env's (B, T, H, W, C) rgb.
        tot, n, ch, m = 0.0, 0, 0.0, 0
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(0)
            for obs, act, noise, t in val_batches:
                cond = model.encode_obs(obs, eval_mode=True)
                pred = model.noise_pred_net(model.noise_scheduler.add_noise(act, noise, t), t, global_cond=cond)
                tot += F.mse_loss(pred, noise, reduction="sum").item()
                n += noise.numel()
                a = model.get_action(dict(obs, rgb=obs["rgb"].permute(0, 1, 3, 4, 2)))
                s = args.obs_horizon - 1
                ch += (a - act[:, s:s + a.shape[1]]).norm(dim=(1, 2)).sum().item()
                m += len(a)
        return ch / m, tot / n

    val_curve, best_val = [], (float("inf"), -1)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)
    save_iters = {args.total_iters - 10000, args.total_iters - 5000, args.total_iters - 1}

    def save_ckpt(tag):
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        ema.copy_to(ema_agent.parameters())
        torch.save({"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict()}, f"runs/{run_name}/checkpoints/{tag}.pt")

    def evaluate_and_save_best(iteration, n_episodes=args.num_eval_episodes):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(n_episodes, ema_agent, envs, device, args.sim_backend)
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            for k in ["success_once", "success_at_end"]:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(f"best_eval_{k}")
                    print(f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.")

    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        last_tick = time.time()
        action_seq = data_batch["actions"]  # (B, pred_horizon, A)
        dp_loss = agent.compute_loss(obs_seq=data_batch["observations"], action_seq=action_seq)
        B = action_seq.shape[0]
        z = feats["z"].reshape(B, args.obs_horizon, -1)[:, -1]  # (B, 256), current frame
        with torch.no_grad():
            s = args.obs_horizon - 1  # executed chunk starts at the current frame
            y = assign(((action_seq[:, s:s + CHUNK] - a_mean) / a_std).reshape(B, -1), centroids)
        sc = supcon(z, y, args.supcon_tau)
        total_loss = dp_loss + args.supcon_weight * sc if args.supcon_weight > 0 else dp_loss
        timings["forward"] += time.time() - last_tick

        if iteration % args.log_freq == 0:
            g_dp = grad_norm(dp_loss, enc_params) if enc_params else 0.0
            g_sc = grad_norm(args.supcon_weight * sc, enc_params) if args.supcon_weight > 0 else 0.0
            for k, v in dict(dp=dp_loss.item(), supcon=sc.item(), grad_enc_dp=g_dp, grad_enc_supcon=g_sc).items():
                writer.add_scalar(f"losses/{k}", v, iteration)
            print(f"iter {iteration}: dp {dp_loss.item():.5f} supcon {sc.item():.4f} | encoder grad norm: "
                  f"dp {g_dp:.4g} supcon(x{args.supcon_weight}) {g_sc:.4g}", flush=True)

        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        timings["backward"] += time.time() - last_tick

        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        if val_batches and (iteration % args.val_freq == 0 or iteration == args.total_iters - 1):
            ema.copy_to(ema_agent.parameters())
            v, v_dp = val_loss(ema_agent)
            val_curve.append((iteration, v, v_dp))
            writer.add_scalar("losses/val_chunk", v, iteration)
            writer.add_scalar("losses/val_dp", v_dp, iteration)
            if v < best_val[0]:
                best_val = (v, iteration)
                os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
                torch.save({"ema_agent": ema_agent.state_dict(), "iteration": iteration, "val_chunk": v},
                           f"runs/{run_name}/checkpoints/best_val.pt")

        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        if iteration in save_iters:
            save_ckpt(str(iteration))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters, args.final_eval_episodes or args.num_eval_episodes)
    log_metrics(args.total_iters)

    if val_batches:  # evaluate the early-stopped checkpoint
        ema_agent.load_state_dict(torch.load(f"runs/{run_name}/checkpoints/best_val.pt", map_location=device)["ema_agent"])
        m = evaluate(args.final_eval_episodes or args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
        m = {k: float(np.mean(v)) for k, v in m.items()}
        print(f"best_val (iteration {best_val[1]}, chunk error {best_val[0]:.5f}): " + " ".join(f"{k} {v:.4f}" for k, v in m.items()))
        with open(f"runs/{run_name}/best_val.json", "w") as f:
            json.dump(dict(iteration=best_val[1], val_chunk=best_val[0], eval=m, val_curve_iter_chunk_dp=val_curve), f, indent=1)

    envs.close()
    writer.close()
