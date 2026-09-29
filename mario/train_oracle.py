# Step 0: train Mario 1-1 oracles in one ROM version, with plain BC and with SCIL (BC + SupCon on
# the embedding e), 3 seeds each. Reports held-out action accuracy and closed-loop distance / flags.
# Run: uv run --project mario python mario/train_oracle.py [rom_version] [epochs]
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import CLASS_NAMES, N_CLASSES, TRAIN_EPS, VAL_EPS, load_episodes
from evaluate import rollout
from models import Controller, Encoder, supcon, to_input

ROM_VERSION = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 standard, 1 no background, 2 pixel
METHODS = ["bc", "scil"]
SEEDS = [0, 1, 2]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # 10 = reference (Kanervisto et al.)
BATCH = 256  # reference uses 32; SCIL recommends large batches so rare classes get positives
LR = 1e-3  # Adam default, as in the reference
L2 = 1e-5  # reference weight decay
LAMBDA_SUPCON = 1.0  # not reported in the SCIL paper
TAU = 0.07  # SCIL appendix code (temperature = base_temperature = 0.07)
EVAL_EPISODES = 10
DEVICE = torch.device("cuda")
OUT = Path(__file__).parent.parent / "results" / (f"{date.today():%Y%m%d}_step0_oracles_v{ROM_VERSION}"
                                                 + ("" if EPOCHS == 10 else f"_e{EPOCHS}"))


@torch.no_grad()
def accuracy(enc, ctrl, x, y):
    pred = torch.cat([ctrl(enc(to_input(x[i:i + 1024]))).argmax(1) for i in range(0, len(x), 1024)])
    per_class = [(pred[y == c] == c).float().mean().item() for c in range(N_CLASSES) if (y == c).any()]
    return (pred == y).float().mean().item(), float(np.mean(per_class))


def train(method, seed, x, y):
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc, ctrl = Encoder().to(DEVICE), Controller(N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam([*enc.parameters(), *ctrl.parameters()], lr=LR, weight_decay=L2)
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(x), device=DEVICE)
        tot = {"ce": 0.0, "supcon": 0.0}
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            e = enc(to_input(x[idx]))
            ce = F.cross_entropy(ctrl(e), y[idx])
            sc = supcon(e, y[idx])
            loss = ce + LAMBDA_SUPCON * sc if method == "scil" else ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["ce"] += ce.item() * len(idx)
            tot["supcon"] += sc.item() * len(idx)  # logged for BC too, to compare cluster quality
        print(f"  epoch {epoch + 1}: ce {tot['ce'] / len(x):.3f} supcon {tot['supcon'] / len(x):.3f}")
    return enc.eval(), ctrl.eval()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr, _, _ = load_episodes(TRAIN_EPS, ROM_VERSION)
    x_va, y_va, _, _ = load_episodes(VAL_EPS, ROM_VERSION)
    x_tr, y_tr, x_va, y_va = x_tr.to(DEVICE), y_tr.to(DEVICE), x_va.to(DEVICE), y_va.to(DEVICE)
    print(f"train {len(x_tr)} / val {len(x_va)} samples | train classes "
          f"{dict(zip(CLASS_NAMES, torch.bincount(y_tr, minlength=N_CLASSES).tolist()))}")

    config = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    metrics = {}
    for method in METHODS:
        for seed in SEEDS:
            print(f"{method} seed {seed}")
            t0 = time.time()
            enc, ctrl = train(method, seed, x_tr, y_tr)
            acc, bal_acc = accuracy(enc, ctrl, x_va, y_va)
            max_x, flags = rollout(lambda x: ctrl(enc(to_input(x))), ROM_VERSION, EVAL_EPISODES, seed, DEVICE)
            metrics[f"{method}_s{seed}"] = {
                "val_acc": acc, "val_balanced_acc": bal_acc,
                "max_x": max_x.tolist(), "flag_rate": float(flags.mean()), "minutes": (time.time() - t0) / 60,
            }
            print(f"  val acc {acc:.3f} (balanced {bal_acc:.3f}) | max x {max_x.mean():.0f} ± {max_x.std():.0f} "
                  f"| flags {flags.mean():.0%}")
            torch.save({"encoder": enc.state_dict(), "controller": ctrl.state_dict()},
                       OUT / f"{method}_s{seed}.pt")

    for method in METHODS:
        m = [metrics[f"{method}_s{s}"] for s in SEEDS]
        f = lambda k: np.array([r[k] if np.isscalar(r[k]) else np.mean(r[k]) for r in m])
        metrics[method] = {k: [float(f(k).mean()), float(f(k).std())]
                           for k in ["val_acc", "val_balanced_acc", "max_x", "flag_rate"]}
        print(f"{method}: " + ", ".join(f"{k} {v[0]:.3f} ± {v[1]:.3f}" for k, v in metrics[method].items()))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
