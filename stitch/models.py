"""Encoder E: pixels -> z and Controller C: [z, proprio] -> action. Plain nn.Modules."""
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
