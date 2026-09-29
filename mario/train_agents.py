# Train agents for any encoder type (nature / resnet18 / dinov2), level and ROM version, with the Step 0
# recipe (50 epochs). Reports held-out accuracy and in-game distance / flags. Methods:
#   bc        cross-entropy only
#   scil      + SupCon directly on the latent e (as in the SCIL paper)
#   scilproj  + SupCon on a projection head MLP(e) (standard SupCon); the controller still reads e and the
#             head is discarded after training
#   tacoK     + TACO temporal InfoNCE (Zheng et al. 2023) with K decisions: [e_t, actions t..t+K-1] <-> e_{t+K},
#             on its own small heads (K = 1 or 3, e.g. taco3)
#   scil_tacoK  SCIL + TACO
# Run: OMP_NUM_THREADS=1 uv run --project mario python mario/train_agents.py <arch> <level> <version> [methods]
#   methods: comma-separated, default bc,scil; results are added to the folder's existing metrics.json
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents import DEVICE, SIZE, embed, encode, game_policy, inputs, new_model, oracle_dir
from data import CLASS_NAMES, EPISODES, N_CLASSES, SKIP
from evaluate import rollout
from models import LATENT, supcon

ARCH, LEVEL, ROM_VERSION = sys.argv[1], sys.argv[2], int(sys.argv[3])
METHODS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["bc", "scil"]
SEEDS = [0, 1, 2]
EPOCHS = 50
BATCH = 256
LR = 1e-3
L2 = 1e-5
LAMBDA_SUPCON = 1.0
LAMBDA_TACO = 1.0
TACO_BATCH = 1024  # transitions per step for the TACO term (its own batch; CE / SupCon keep BATCH)
TACO_TAU = 0.1
EVAL_EPISODES = 10
OUT = oracle_dir(ARCH, LEVEL, ROM_VERSION)


class TacoHeads(nn.Module):
    # TACO heads: query = MLP([e_t, emb(a_t), ..., emb(a_{t+K-1})]), key = MLP(e_{t+K}), both 128-d
    def __init__(self, k):
        super().__init__()
        self.act = nn.Embedding(N_CLASSES, 32)
        self.query = nn.Sequential(nn.Linear(LATENT + 32 * k, 256), nn.ReLU(), nn.Linear(256, 128))
        self.key = nn.Sequential(nn.Linear(LATENT, 256), nn.ReLU(), nn.Linear(256, 128))

    def loss(self, e_t, acts, e_tk):
        # e_t, e_tk: (B, 512) latents K decisions apart; acts: (B, K) action classes in between.
        # InfoNCE: the true future is the positive, the other futures in the batch are negatives.
        q = F.normalize(self.query(torch.cat([e_t, self.act(acts).flatten(1)], 1)), dim=1)
        k = F.normalize(self.key(e_tk), dim=1)
        return F.cross_entropy(q @ k.T / TACO_TAU, torch.arange(len(q), device=q.device))


def train(method, seed, X, y, ep):
    torch.manual_seed(seed)
    enc, ctrl = new_model(ARCH)
    params = [*enc.parameters(), *ctrl.parameters()]
    if method == "scilproj":  # projection head (SupCon paper: MLP with 128-d output); created only here so
        # the other methods' random streams, and hence their results, stay exactly reproducible
        head = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, 128)).to(DEVICE)
        params += [*head.parameters()]
    K = int(method[-1]) if "taco" in method else 0
    if K:
        taco = TacoHeads(K).to(DEVICE)
        params += [*taco.parameters()]
        gap = SKIP * K  # K decisions = SKIP * K recorded frames
        starts = torch.nonzero(ep[:-gap] == ep[gap:]).squeeze(1)  # frame t and t + gap in the same episode
        offsets = torch.arange(K, device=DEVICE) * SKIP  # the K decisions between t and t + gap
    use_supcon = method == "scil" or method.startswith("scil_taco")
    opt = torch.optim.Adam(params, lr=LR, weight_decay=L2)
    hist = []  # per-epoch mean losses, to spot conflicts between the terms
    for epoch in range(EPOCHS):
        tot = {"ce": 0.0, "supcon": 0.0, "taco": 0.0}
        n = 0
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            e = encode(ARCH, enc, X[idx])
            ce = F.cross_entropy(ctrl(e), y[idx])
            loss = ce
            tot["ce"] += ce.item()
            if use_supcon:
                sc = supcon(e, y[idx])
                loss = loss + LAMBDA_SUPCON * sc
                tot["supcon"] += sc.item()
            if method == "scilproj":
                loss = loss + LAMBDA_SUPCON * supcon(head(e), y[idx])
            if K:
                t = starts[torch.randint(len(starts), (TACO_BATCH,), device=DEVICE)]
                tc = taco.loss(encode(ARCH, enc, X[t]), y[t[:, None] + offsets], encode(ARCH, enc, X[t + gap]))
                loss = loss + LAMBDA_TACO * tc
                tot["taco"] += tc.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            n += 1
        hist.append({k: v / n for k, v in tot.items()})
        if K:
            print(f"  {method} s{seed} epoch {epoch + 1}: " + " ".join(f"{k} {v:.3f}" for k, v in hist[-1].items()), flush=True)
    return enc.eval(), ctrl.eval(), hist


@torch.no_grad()
def balanced_acc(enc, ctrl, X, y):
    pred = ctrl(torch.from_numpy(embed(ARCH, enc, X)).float().to(DEVICE)).argmax(1).cpu()
    return float((pred == y).float().mean()), float(np.mean([(pred[y == c] == c).float().mean() for c in y.unique()]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_eps, val_eps = EPISODES[LEVEL]
    X, y, ep = inputs(ARCH, LEVEL, ROM_VERSION, train_eps)
    X_va, y_va, _ = inputs(ARCH, LEVEL, ROM_VERSION, val_eps)
    X, y, ep = X.to(DEVICE), y.to(DEVICE), ep.to(DEVICE)
    print(f"{ARCH} {LEVEL} v{ROM_VERSION}: train {len(X)} / val {len(X_va)} | train classes "
          f"{dict(zip(CLASS_NAMES, torch.bincount(y, minlength=N_CLASSES).tolist()))}", flush=True)
    config = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
    tag = "" if METHODS == ["bc", "scil"] else "_" + "_".join(METHODS)
    (OUT / f"config{tag}.json").write_text(json.dumps({**config, "train_eps": train_eps, "val_eps": val_eps}, indent=2))

    metrics = json.loads((OUT / "metrics.json").read_text()) if (OUT / "metrics.json").exists() else {}
    for method in METHODS:
        for seed in SEEDS:
            t0 = time.time()
            enc, ctrl, hist = train(method, seed, X, y, ep)
            acc, bal = balanced_acc(enc, ctrl, X_va, y_va)
            max_x, flags = rollout(game_policy(ARCH, enc, ctrl), ROM_VERSION, EVAL_EPISODES, seed, DEVICE,
                                   LEVEL, SIZE[ARCH])
            metrics[f"{method}_s{seed}"] = {"val_acc": acc, "val_balanced_acc": bal, "max_x": max_x.tolist(),
                                            "flag_rate": float(flags.mean()), "minutes": (time.time() - t0) / 60,
                                            "loss_history": hist}
            torch.save({"encoder": enc.state_dict(), "controller": ctrl.state_dict()}, OUT / f"{method}_s{seed}.pt")
            print(f"{method} s{seed}: val bal acc {bal:.3f} | max x {max_x.mean():.0f} ± {max_x.std():.0f} "
                  f"| flags {flags.mean():.0%}", flush=True)
    for method in METHODS:
        m = [metrics[f"{method}_s{s}"] for s in SEEDS]
        f = lambda k: np.array([np.mean(r[k]) for r in m])
        metrics[method] = {k: [float(f(k).mean()), float(f(k).std())]
                           for k in ["val_acc", "val_balanced_acc", "max_x", "flag_rate"]}
        print(f"{method}: " + ", ".join(f"{k} {v[0]:.3f} ± {v[1]:.3f}" for k, v in metrics[method].items()), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
