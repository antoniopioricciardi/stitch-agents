# Aligners. Every aligner returns (R, b) with z_mapped = z @ R.T + b (see CLAUDE.md).
# Plain numpy, float64. Zs: (N, d) source latents, Zt: (M, d) target latents.
import numpy as np


def fit_identity(Zs, Zt, **info):
    d = Zs.shape[1]
    return np.eye(d), np.zeros(d)


def fit_procrustes_paired(Zs, Zt, **info):
    # SAPS: rows of Zs and Zt correspond (N == M). Orthogonal Q minimising
    # ||(Zs - mu_s) Q - (Zt - mu_t)||_F is Q = U V^T with U S V^T = svd((Zs - mu_s)^T (Zt - mu_t)).
    # Then z_mapped = (z - mu_s) Q + mu_t, i.e. R = Q^T, b = mu_t - mu_s Q.
    mu_s, mu_t = Zs.mean(0), Zt.mean(0)
    U, _, Vt = np.linalg.svd((Zs - mu_s).T @ (Zt - mu_t))
    Q = U @ Vt
    return Q.T, mu_t - mu_s @ Q


def centroids(Z, y, classes):
    # (len(classes), d) class means
    return np.stack([Z[y == c].mean(0) for c in classes])


def action_pairs(ys, yt, rng, pool=None, per_class=100):
    # Anchor pairs without paired frames (old-repo recipe, the method's default): optionally a random pool
    # of `pool` frames per side, then for each action present on both sides, random frames sharing it are
    # paired, up to per_class pairs. Returns index arrays (ia into the source, ib into the target).
    s = rng.permutation(len(ys))[:pool] if pool is not None else np.arange(len(ys))
    t = rng.permutation(len(yt))[:pool] if pool is not None else np.arange(len(yt))
    ia, ib = [], []
    for c in np.intersect1d(ys[s], yt[t]):
        a, b = rng.permutation(s[ys[s] == c]), rng.permutation(t[yt[t] == c])
        n = min(len(a), len(b), per_class)
        ia.append(a[:n])
        ib.append(b[:n])
    return np.concatenate(ia), np.concatenate(ib)


def fit_action_pairs(Zs, Zt, ys, yt, rng, pool=None, per_class=100, **info):
    # Paired Procrustes on same-action anchor pairs. Within an action the pairing is arbitrary: on average
    # this is centroid Procrustes weighted by pair counts, plus noise from the within-action pairings.
    ia, ib = action_pairs(ys, yt, rng, pool, per_class)
    return fit_procrustes_paired(Zs[ia], Zt[ib])


def fit_prototypes(Zs, Zt, ys, yt, **info):
    # Unpaired: Zs and Zt need not correspond. Class centroids (action prototypes) are the anchors,
    # then paired Procrustes on the K centroid pairs. The rotation is only determined on the <= K-1
    # dimensional span of the centred centroids; outside it the SVD picks an arbitrary completion.
    classes = np.intersect1d(np.unique(ys), np.unique(yt))
    return fit_procrustes_paired(centroids(Zs, ys, classes), centroids(Zt, yt, classes))
