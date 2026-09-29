# Mnih2015 network as used by the SCIL Atari baseline (Kanervisto et al. 2020, utils/networks.py),
# split at the SCIL embedding e (fc1 output, 512-d after ReLU) into encoder and controller, so that
# a stitch is controller_v(T(encoder_u(x))).
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 512


class Encoder(nn.Module):
    # x: (N, 3, 84, 84) float in [0, 1] -> e: (N, 512)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, LATENT)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.fc1(x.flatten(1)))


class Controller(nn.Module):
    # e: (N, 512) -> logits: (N, n_actions). The reference head is a single linear layer.
    def __init__(self, n_actions):
        super().__init__()
        self.fc2 = nn.Linear(LATENT, n_actions)

    def forward(self, e):
        return self.fc2(e)


def to_input(x_uint8):
    # (N, 3, 84, 84) uint8 -> float in [0, 1], as in the reference (/255)
    return x_uint8.float() / 255.0


def supcon(e, y, tau=0.07):
    # SupCon loss from the SCIL appendix, with its fix for anchors that have no positive in the batch
    # (they contribute 0). e: (N, d) embeddings, y: (N,) class labels.
    z = F.normalize(e, dim=1)
    logits = z @ z.T / tau
    logits = logits - logits.max(1, keepdim=True).values.detach()  # numerical stability
    not_self = ~torch.eye(len(y), dtype=torch.bool, device=y.device)
    pos = (y[:, None] == y[None, :]) & not_self
    log_prob = logits - torch.log((logits.exp() * not_self).sum(1, keepdim=True) + 1e-12)
    n_pos = pos.sum(1).clamp(min=1)
    return -((log_prob * pos).sum(1) / n_pos).mean()
