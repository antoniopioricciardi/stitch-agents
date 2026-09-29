# The three encoder types, behind the same few functions so training / stitching scripts can take
# an ARCH constant:
#   nature    Mnih2015 CNN trained from scratch on 84x84 merged frames (models.Encoder)
#   resnet18  frozen ImageNet ResNet18 (512-d pooled features) + trainable fc layer -> e
#   dinov2    frozen DINOv2 ViT-S/14 (384-d CLS features) + trainable fc layer -> e
# For the frozen ones, the trainable fc + ReLU is the "green layer" of the SCIL figure: SupCon is on
# its output e, and the controller (one linear layer) reads e. The frozen backbone sees 224x224 frames
# (same 2-frame max merge, ImageNet normalisation); its features are cached per episode on disk.
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import DATA, N_CLASSES, load_episodes
from models import LATENT, Controller, Encoder, to_input

ROOT = Path(__file__).resolve().parent.parent / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FROZEN = {"resnet18": 512, "dinov2": 384}  # backbone feature dim
SIZE = {"nature": 84, "resnet18": 224, "dinov2": 224}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
_backbones = {}


class FrozenHead(nn.Module):
    # frozen backbone features (N, D) -> e (N, 512): the trainable green layer
    def __init__(self, d_in):
        super().__init__()
        self.fc = nn.Linear(d_in, LATENT)

    def forward(self, f):
        return F.relu(self.fc(f))


def backbone(arch):
    # frozen pretrained backbone, loaded once: (N, 3, 224, 224) normalised -> (N, D)
    if arch not in _backbones:
        if arch == "resnet18":
            import torchvision
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            m.fc = nn.Identity()
        else:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _backbones[arch] = m.to(DEVICE).eval().requires_grad_(False)
    return _backbones[arch]


@torch.no_grad()
def features(arch, x):
    # x: (N, 3, 224, 224) uint8 merged frames -> (N, D) float backbone features
    out = []
    for i in range(0, len(x), 256):
        xb = (x[i:i + 256].to(DEVICE).float() / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        out.append(backbone(arch)(xb).float())
    return torch.cat(out)


def inputs(arch, level, version, eps):
    # Model inputs for these episodes: merged 84x84 uint8 frames (nature) or cached backbone features
    # (frozen). Returns X (N, ...) on CPU, y (N,), ep (N,) — rows aligned across versions (replays).
    if arch == "nature":
        x, y, e, _ = load_episodes(eps, version, level)
        return x, y, e
    Xs, ys, es = [], [], []
    for ep in eps:
        f = DATA / "feats" / arch / f"{level}_v{version}_ep{ep:03d}.pt"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            x, y, e, _ = load_episodes([ep], version, level, size=224)
            torch.save({"X": features(arch, x).half().cpu(), "y": y, "e": e}, f)
        d = torch.load(f)
        Xs.append(d["X"].float()), ys.append(d["y"]), es.append(d["e"])
    return torch.cat(Xs), torch.cat(ys), torch.cat(es)


def new_model(arch):
    enc = Encoder() if arch == "nature" else FrozenHead(FROZEN[arch])
    return enc.to(DEVICE), Controller(N_CLASSES).to(DEVICE)


def encode(arch, enc, X):
    # model inputs (on DEVICE) -> e; the nature encoder takes uint8 frames
    return enc(to_input(X)) if arch == "nature" else enc(X)


def oracle_dir(arch, level, version):
    if arch == "nature" and level == "1-1":
        return ROOT / f"20260929_step0_oracles_v{version}_e50"  # the Step 0 oracles
    return ROOT / f"20260929_oracles_{arch}_{level}_v{version}"


def load_model(arch, method, seed, level, version):
    enc, ctrl = new_model(arch)
    ck = torch.load(oracle_dir(arch, level, version) / f"{method}_s{seed}.pt", map_location=DEVICE)
    enc.load_state_dict(ck["encoder"])
    ctrl.load_state_dict(ck["controller"])
    return enc.eval(), ctrl.eval()


@torch.no_grad()
def embed(arch, enc, X):
    # (N, ...) CPU inputs -> (N, 512) float64 numpy latents
    return torch.cat([encode(arch, enc, X[i:i + 1024].to(DEVICE))
                      for i in range(0, len(X), 1024)]).double().cpu().numpy()


def game_policy(arch, enc, ctrl, R=None, b=None):
    # policy for evaluate.rollout (merged uint8 frames at SIZE[arch] -> logits), optionally with a
    # stitching map z -> z @ R.T + b (R, b torch tensors on DEVICE)
    def policy(x):
        X = x if arch == "nature" else features(arch, x)
        e = encode(arch, enc, X)
        return ctrl(e if R is None else e @ R.T + b)
    return policy
