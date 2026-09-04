"""
Weighted Rank Clique Filtration + Dirac spectral geometry on networks.
======================================================================

Implements, from the mathematics in my talk "Triadic Topological Brain
Networks" (Joshi, City St George's, University of London):

  1. WRCF: rank edges by weight w1 >= w2 >= ... >= wm, assign filtration
     values f(e_k) = k/m, and build clique complexes K(f) where a simplex
     enters when all of its edges have entered:  f(sigma) = max_{e in sigma} f(e).
  2. Signed boundary operators B1 (edges->nodes) and B2 (triangles->edges)
     with consistent orientation.
  3. The Dirac operator on the 0-1-2 complex,

         D = [ 0    B1^T   0   ]
             [ B1    0    B2^T ]
             [ 0    B2     0   ]

     with D^2 = diag(L0, L1, L2): with our convention B1: C1->C0, the Hodge
     Laplacians are L0 = B1 B1^T, L1 = B1^T B1 + B2 B2^T, L2 = B2^T B2.
     The code verifies D^2 == diag(L0, L1, L2) numerically at runtime.
  4. The Dirac gap  gamma = min{ |lambda| : lambda in spec(D), lambda != 0 },
     i.e. gamma^2 is the smallest non-zero Hodge eigenvalue -- the weakest
     supported topological mode.

Experiment: a "control" network rich in strong triangles vs a "degraded"
network whose triadic closure is weakened. We track gamma(tau) across the
filtration and show the degraded network's spectral collapse -- an effect
invisible to edge-level summaries with matched density.

Run:  python run_demo.py     (writes dirac_spectrum.png, gamma_curve.png)
"""
import os
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

rng = np.random.default_rng(11)


# ------------------------------------------------------------- networks ----
def community_network(n=30, k=3, p_in=0.9, p_out=0.08, triangle_boost=0.35):
    """Weighted network with strong within-community triads ('control')."""
    comm = np.repeat(np.arange(k), n // k)
    W = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        p = p_in if comm[i] == comm[j] else p_out
        if rng.random() < p:
            w = rng.uniform(0.4, 1.0) if comm[i] == comm[j] else rng.uniform(0.05, 0.35)
            W[i, j] = W[j, i] = w
    # reinforce triangles inside communities
    for i, j, l in combinations(range(n), 3):
        if comm[i] == comm[j] == comm[l] and W[i, j] and W[j, l] and W[i, l]:
            for a, b in [(i, j), (j, l), (i, l)]:
                W[a, b] = W[b, a] = min(1.0, W[a, b] + triangle_boost * rng.random())
    return W


def degrade_triangles(W, frac=0.6):
    """Weaken one edge of a fraction of triangles: same nodes, similar edge
    mass, far fewer *strong closed* triads ('degraded')."""
    W = W.copy()
    tris = [t for t in combinations(range(len(W)), 3)
            if W[t[0], t[1]] and W[t[1], t[2]] and W[t[0], t[2]]]
    rng.shuffle(tris)
    for i, j, l in tris[: int(frac * len(tris))]:
        pairs = [(i, j), (j, l), (i, l)]
        a, b = pairs[np.argmin([W[a, b] for a, b in pairs])]
        W[a, b] = W[b, a] = W[a, b] * 0.15
    return W


# ----------------------------------------------------------------- WRCF ----
def wrcf_complex(W, tau):
    """Clique complex at filtration value tau under WRCF."""
    n = len(W)
    iu = np.triu_indices(n, 1)
    weights = W[iu]
    order = np.argsort(-weights)                       # rank by weight, desc
    m = (weights > 0).sum()
    fvals = np.full(len(weights), np.inf)
    fvals[order[:m]] = (np.arange(m) + 1) / m          # f(e_k) = k/m
    keep = fvals <= tau
    edges = sorted((iu[0][t], iu[1][t]) for t in np.where(keep)[0])
    eset = set(edges)
    triangles = sorted((i, j, l) for i, j, l in combinations(range(n), 3)
                       if (i, j) in eset and (j, l) in eset and (i, l) in eset)
    return list(range(n)), edges, triangles


# ------------------------------------------- boundary and Dirac operators --
def boundary_matrices(nodes, edges, triangles):
    """Signed incidence matrices with the standard orientation (sorted
    vertex order): d[i,j] = j - i;  d[i,j,k] = [j,k] - [i,k] + [i,j]."""
    eidx = {e: c for c, e in enumerate(edges)}
    B1 = np.zeros((len(nodes), len(edges)))
    for c, (i, j) in enumerate(edges):
        B1[i, c], B1[j, c] = -1.0, 1.0
    B2 = np.zeros((len(edges), len(triangles)))
    for c, (i, j, k) in enumerate(triangles):
        B2[eidx[(j, k)], c] += 1.0
        B2[eidx[(i, k)], c] -= 1.0
        B2[eidx[(i, j)], c] += 1.0
    return B1, B2


def dirac_operator(B1, B2):
    n0, n1 = B1.shape
    n2 = B2.shape[1]
    D = np.zeros((n0 + n1 + n2, n0 + n1 + n2))
    D[:n0, n0:n0 + n1] = B1
    D[n0:n0 + n1, :n0] = B1.T
    D[n0:n0 + n1, n0 + n1:] = B2
    D[n0 + n1:, n0:n0 + n1] = B2.T
    return D


def dirac_gap(D, tol=1e-8):
    lam = np.linalg.eigvalsh(D)
    nz = np.abs(lam)[np.abs(lam) > tol]
    return (nz.min() if len(nz) else 0.0), lam


def sanity_check_squares(B1, B2, D):
    """Verify D^2 = diag(L0, L1, L2)."""
    n0, n1 = B1.shape
    L0 = B1 @ B1.T
    L1 = B1.T @ B1 + B2 @ B2.T
    L2 = B2.T @ B2
    block = np.zeros_like(D)
    block[:n0, :n0] = L0
    block[n0:n0 + n1, n0:n0 + n1] = L1
    block[n0 + n1:, n0 + n1:] = L2
    assert np.allclose(D @ D, block, atol=1e-10), "D^2 != diag(Hodge Laplacians)"


if __name__ == "__main__":
    Wc = community_network()
    Wd = degrade_triangles(Wc)

    # ---- spectrum at a fixed filtration scale --------------------------
    TAU = 0.5
    spectra, gaps, tri_counts = {}, {}, {}
    for name, W in [("control", Wc), ("degraded", Wd)]:
        nodes, edges, tris = wrcf_complex(W, TAU)
        B1, B2 = boundary_matrices(nodes, edges, tris)
        D = dirac_operator(B1, B2)
        sanity_check_squares(B1, B2, D)
        g, lam = dirac_gap(D)
        spectra[name], gaps[name], tri_counts[name] = lam, g, len(tris)
        print(f"{name:9s} tau={TAU}: |edges|={len(edges):3d} |triangles|={len(tris):3d} "
              f"Dirac gap gamma={g:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (name, lam) in zip(axes, spectra.items()):
        ax.plot(np.sort(lam), ".", ms=3, color="#1F3864")
        ax.axhline(0, c="k", lw=0.5)
        ax.set_title(f"{name}: Dirac spectrum (gamma={gaps[name]:.3f})")
        ax.set_xlabel("index")
    axes[0].set_ylabel("eigenvalue")
    plt.tight_layout(); plt.savefig("dirac_spectrum.png", dpi=150)

    # ---- gamma across the filtration -----------------------------------
    taus = np.linspace(0.15, 0.9, 16)
    plt.figure(figsize=(7, 4.2))
    for name, W in [("control", Wc), ("degraded", Wd)]:
        gs = []
        for t in taus:
            nodes, edges, tris = wrcf_complex(W, t)
            B1, B2 = boundary_matrices(nodes, edges, tris)
            g, _ = dirac_gap(dirac_operator(B1, B2))
            gs.append(g)
        plt.plot(taus, gs, "o-", label=name,
                 color="#1F3864" if name == "control" else "#C00000")
    plt.xlabel("WRCF filtration value tau"); plt.ylabel("Dirac gap gamma")
    plt.title("Spectral collapse under triadic degradation")
    plt.legend(); plt.tight_layout(); plt.savefig("gamma_curve.png", dpi=150)
    print("wrote dirac_spectrum.png, gamma_curve.png")
