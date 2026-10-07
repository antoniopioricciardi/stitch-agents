"""Encoder E: pixels -> z and Controller C: [z, proprio] -> action. Plain nn.Modules."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # obs (N, 128, 128, 3) uint8 -> z (N, d). Four stride-2 convs (128 -> 8), then a linear map;
    # flatten (not global pooling) keeps where things are, which the controller needs.
    def __init__(self, d=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 8 * 8, d)

    def forward(self, obs):
        x = obs.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.fc(self.convs(x).flatten(1))


class Controller(nn.Module):
    # [z (N, d), proprio (N, P)] -> action chunk (N, chunk, a), in real action units.
    # With history, z and proprio are the concatenation over the last frames (the caller stacks them).
    # Proprio and actions are standardised per dimension with training-set mean/std (buffers, so they travel
    # with the weights): the MLP predicts standardised actions and forward() maps them back.
    def __init__(self, d=256, p=25, a=7, hidden=256, chunk=1):
        super().__init__()
        self.chunk, self.a = chunk, a
        self.register_buffer("p_mean", torch.zeros(p))
        self.register_buffer("p_std", torch.ones(p))
        self.register_buffer("a_mean", torch.zeros(a))
        self.register_buffer("a_std", torch.ones(a))
        self.mlp = nn.Sequential(
            nn.Linear(d + p, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, a * chunk),
        )

    def forward(self, z, proprio):
        out = self.mlp(torch.cat([z, (proprio - self.p_mean) / self.p_std], 1)).view(-1, self.chunk, self.a)
        return out * self.a_std + self.a_mean


def random_shift(obs, pad=4):
    # DrQ augmentation: pad by replicating edges, then crop back at a random offset per sample.
    # obs (N, H, W, 3) uint8 -> same shape, uint8
    n, h, w, _ = obs.shape
    x = F.pad(obs.permute(0, 3, 1, 2).float(), (pad,) * 4, mode="replicate")
    dx, dy = torch.randint(0, 2 * pad + 1, (2, n), device=obs.device)
    rows = (torch.arange(h, device=obs.device)[None] + dy[:, None])  # (N, H)
    cols = (torch.arange(w, device=obs.device)[None] + dx[:, None])  # (N, W)
    x = x[torch.arange(n, device=obs.device)[:, None, None, None], torch.arange(3, device=obs.device)[None, :, None, None],
          rows[:, None, :, None], cols[:, None, None, :]]
    return x.permute(0, 2, 3, 1).to(torch.uint8)


DINO_HUB = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")  # local copy, no network at load time


class DinoEncoder(nn.Module):
    # Step 3: drop-in for the DP baseline's PlainConv. rgb (B, 3, 128, 128) in [0, 1] -> z (B, 256).
    # DINOv2 ViT-S/14 on the image resized to `size` px (126: a 9 x 9 grid of 14-px patches; 224: 16 x 16),
    # ImageNet-normalised, in bf16 (RTX 5070 Ti, per batch of 512 frames at 126 px: ~48 ms frozen, ~154 ms fine-tuned;
    # 224 px frozen ~160 ms), then a trainable adapter:
    #   cls:     the CLS token (B, 384) -> Linear -> 256
    #   spatial: the patch grid (B, 384, g, g) -> 1x1 conv to 32 channels, ReLU -> flatten (B, 32 g^2) -> Linear -> 256;
    #            keeps where things are (the cube is only in the image).
    def __init__(self, adapter="cls", frozen=True, d=256, c=32, size=126):
        super().__init__()
        self.backbone = torch.hub.load(DINO_HUB, "dinov2_vits14", source="local", verbose=False)
        self.backbone.requires_grad_(not frozen)
        self.frozen, self.spatial, self.size, self.g = frozen, adapter == "spatial", size, size // 14
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if self.spatial:
            self.adapter = nn.Sequential(nn.Conv2d(384, c, 1), nn.ReLU(), nn.Flatten(), nn.Linear(c * self.g ** 2, d))
        else:
            self.adapter = nn.Linear(384, d)

    def features(self, x):
        # backbone output fed to the adapter: CLS (B, 384) or patch grid (B, 384, g, g), float32
        x = (F.interpolate(x, size=self.size, mode="bilinear", antialias=True) - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            out = self.backbone.forward_features(x)
        if self.spatial:
            return out["x_norm_patchtokens"].float().transpose(1, 2).reshape(len(x), 384, self.g, self.g)
        return out["x_norm_clstoken"].float()

    def forward(self, x):
        return self.adapter(self.features(x))


def dino_from_sd(sd):
    # an empty DinoEncoder matching an agent state dict (Step 3). Spatial adapter: the input size follows from the grid,
    # Linear in = 32 g^2 (CLS checkpoints are all 126 px).
    spatial = "visual_encoder.adapter.0.weight" in sd
    g = round((sd["visual_encoder.adapter.3.weight"].shape[1] / 32) ** 0.5) if spatial else 9
    return DinoEncoder("spatial" if spatial else "cls", size=14 * g).to(sd["visual_encoder.mean"].device)


def swap_dino(agent, sd):
    # before agent.load_state_dict(sd): if the checkpoint has a DINO encoder (Step 3), put one in place of PlainConv
    if "visual_encoder.backbone.cls_token" in sd:
        agent.visual_encoder = dino_from_sd(sd)
    return agent
