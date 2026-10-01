# Action-chunk labels for continuous control (Step 1b): frame t -> k-means cluster of its executed DP chunk.
# Chunks follow the DP baseline's slicing (train_rgbd.SmallDemoDataset_DiffusionPolicy, obs horizon 2,
# pred horizon 16): frame t is the current frame of the slice starting at t-1, for t = 0 .. L-2, and its executed
# actions are act_seq[1:9] = a[t : t+8]. Steps past the end are padded as the baseline does: zero arm action,
# the last gripper action.
import numpy as np
import torch

CHUNK = 8  # = act_horizon


def frame_chunks(actions):
    # actions: (L, A) one demo -> (L-1, CHUNK, A) chunks for frames t = 0 .. L-2
    L, A = actions.shape
    pad = np.zeros((CHUNK, A), actions.dtype)
    pad[:, -1] = actions[-1, -1]
    a = np.concatenate([actions, pad])
    return np.stack([a[t:t + CHUNK] for t in range(L - 1)])


def standardise(chunks, mean, std):
    # (N, CHUNK, A) -> (N, CHUNK*A); z-score per action dimension (mean, std: (A,) over the training actions),
    # so the gripper's +-1 does not dominate the distances
    return ((chunks - mean) / std).reshape(len(chunks), -1)


def kmeans(X, K, restarts=10, iters=300, seed=0):
    # X: (N, D) float tensor -> centroids (K, D) of the restart with the lowest inertia. k-means++ init, Lloyd.
    g = torch.Generator(device=X.device).manual_seed(seed)
    best, best_inertia = None, float("inf")
    for _ in range(restarts):
        C = X[torch.randint(len(X), (1,), generator=g, device=X.device)]
        for _ in range(K - 1):  # k-means++: next centre with probability ~ squared distance to the nearest one
            d2 = torch.cdist(X, C).min(1).values ** 2
            C = torch.cat([C, X[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
        for _ in range(iters):
            y = torch.cdist(X, C).argmin(1)
            C_new = torch.stack([X[y == k].mean(0) if (y == k).any() else C[k] for k in range(K)])
            if torch.allclose(C_new, C):
                break
            C = C_new
        inertia = (torch.cdist(X, C).min(1).values ** 2).sum().item()
        if inertia < best_inertia:
            best, best_inertia = C, inertia
    return best, best_inertia


def assign(X, C):
    # X: (N, D), C: (K, D) -> (N,) nearest-centroid labels
    return torch.cdist(X, C).argmin(1)
