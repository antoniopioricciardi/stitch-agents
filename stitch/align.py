# Aligners. Every aligner returns (R, b) with z_mapped = z @ R.T + b (see CLAUDE.md).
# Plain numpy, float64. Zs: (N, d) source latents, Zt: (M, d) target latents.
# fit_identity, fit_procrustes_paired, action_pairs, fit_action_pairs: copied unchanged from mario/align.py.
import numpy as np
import ot


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


# --- Gromov-Wasserstein aligners ---
# GW on all training frames needs N x N cost matrices, so both fit on a subsample of n frames per side,
# then the map (fitted on the subsample) is applied to all frames.

def structure(Z):
    # (n, d) -> (n, n) pairwise Euclidean distances, scaled to [0, 1] so eps means the same on both sides
    C = ot.dist(Z, Z, metric="euclidean")
    return C / C.max()


def barycentric_procrustes(Zs, Zt, P):
    # Each source frame i is sent to its barycentric target sum_j P_ij Zt_j / sum_j P_ij, then paired
    # Procrustes between the frames and their barycentres. Zs: (n, d), Zt: (m, d), P: (n, m).
    return fit_procrustes_paired(Zs, P @ Zt / P.sum(1, keepdims=True))


def gw_objective(C1, C2, P):
    # square-loss GW term sum_ijkl (C1_ik - C2_jl)^2 P_ij P_kl, without the entropy, using the plan's actual
    # marginals p = P 1, q = P^T 1: p^T C1^2 p + q^T C2^2 q - 2 <C1 P C2, P>
    p, q = P.sum(1), P.sum(0)
    return float(p @ C1**2 @ p + q @ C2**2 @ q - 2 * ((C1 @ P @ C2) * P).sum())


def fit_gw(Zs, Zt, rng, eps, n=1000, diag=None, **info):
    # Pure geometry, label-free: uniform random subsample (not stratified by action), entropic GW between the
    # two intra-space distance matrices with uniform marginals, then barycentric Procrustes.
    # diag (optional dict) receives the plan, subsample indices and final GW objective (label-free).
    i_s, i_t = rng.permutation(len(Zs))[:n], rng.permutation(len(Zt))[:n]
    C1, C2 = structure(Zs[i_s]), structure(Zt[i_t])
    P = ot.gromov.entropic_gromov_wasserstein(C1, C2, epsilon=eps)
    if diag is not None:
        diag.update(P=P, i_s=i_s, i_t=i_t, obj=gw_objective(C1, C2, P))
    return barycentric_procrustes(Zs[i_s], Zt[i_t], P)


def fit_action_pairs_fgw(Zs, Zt, ys, yt, rng, eps, per_class=100, all_labels=True, alpha=0.5, n=1000,
                         diag=None, **info):
    # Action-pair map (R0, b0) as initialisation, then fused GW refinement and barycentric Procrustes.
    # The fused (cross-space) cost tells GW which pairings are plausible:
    #   M = (label mismatch + ||R0 zs + b0 - zt||^2 / max) / 2, both terms in [0, 1]
    # and alpha = 0.5 weights the structure (GW) term against M.
    # Label budget: all_labels=True uses every frame's action (uniform subsample). all_labels=False is the
    # few-pair setting: only the anchor frames keep a label; they are forced into the subsample, and any
    # pair with an unlabelled frame gets a neutral mismatch of 0.5.
    ia, ib = action_pairs(ys, yt, rng, per_class=per_class)  # same draw as fit_action_pairs with this rng
    R0, b0 = fit_procrustes_paired(Zs[ia], Zt[ib])
    if all_labels:
        i_s, i_t = rng.permutation(len(Zs))[:n], rng.permutation(len(Zt))[:n]
        ls, lt = ys[i_s], yt[i_t]
    else:
        i_s = np.concatenate([ia, rng.permutation(np.setdiff1d(np.arange(len(Zs)), ia))[:n - len(ia)]])
        i_t = np.concatenate([ib, rng.permutation(np.setdiff1d(np.arange(len(Zt)), ib))[:n - len(ib)]])
        ls = np.concatenate([ys[ia], np.full(n - len(ia), -1)])  # -1 = unlabelled
        lt = np.concatenate([yt[ib], np.full(n - len(ib), -1)])
    mismatch = (ls[:, None] != lt[None, :]).astype(float)
    mismatch[(ls[:, None] < 0) | (lt[None, :] < 0)] = 0.5
    D = ot.dist(Zs[i_s] @ R0.T + b0, Zt[i_t])  # squared Euclidean
    M = (mismatch + D / D.max()) / 2
    P = ot.gromov.entropic_fused_gromov_wasserstein(M, structure(Zs[i_s]), structure(Zt[i_t]),
                                                    epsilon=eps, alpha=alpha)
    if diag is not None:
        diag.update(P=P, i_s=i_s, i_t=i_t)
    return barycentric_procrustes(Zs[i_s], Zt[i_t], P)
