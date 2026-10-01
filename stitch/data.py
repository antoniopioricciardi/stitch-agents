"""Demo loading helpers: action chunks and observation history over flat (T, ...) arrays with an episode column."""
import numpy as np


def chunk_targets(action, episode, H):
    # action (T, A), episode (T,) -> (T, H, A): actions t..t+H-1 of the same episode. Past the episode end the
    # arm dims are padded with 0 (delta control: stay still) and the gripper with its last value (as ManiSkill's DP).
    T, A = action.shape
    out = np.zeros((T, H, A), dtype=np.float32)
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        a = action[idx]
        pad = np.zeros((H, A), dtype=np.float32)
        pad[:, -1] = a[-1, -1]
        a = np.concatenate([a, pad])
        for k in range(H):
            out[idx, k] = a[k:k + len(idx)]
    return out


def prev_index(episode):
    # (T,) -> index of the previous step in the same episode; the first step points to itself
    prev = np.arange(len(episode)) - 1
    prev[0] = 0
    first = np.r_[True, episode[1:] != episode[:-1]]
    prev[first] = np.where(first)[0]
    return prev
