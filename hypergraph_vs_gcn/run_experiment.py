"""
Why pairwise models cannot see higher-order interactions.
=========================================================

Synthetic experiment from my PhD research proposal (Joshi, City St George's,
University of London): plant a purely *triadic* (XOR-type) interaction among a
group of nodes and ask two models to identify which group carries it.

  * Class 0: nodes {0,1,2} are synergistically coupled (x2 = x0 * x1 signs)
  * Class 1: nodes {3,4,5} are synergistically coupled

The XOR construction guarantees that every PAIRWISE correlation is zero in
expectation -- the interaction exists only at third order. A model restricted
to pairwise structure therefore has provably no usable signal; a model that
aggregates over candidate hyperedges (triples) separates the classes easily.

Both models share the same training procedure (logistic regression trained by
gradient descent); they differ only in the structure their features respect.

Run:  python run_experiment.py       (writes results.png, prints accuracies)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

rng = np.random.default_rng(7)

N_NODES, T = 6, 200
TRIPLES = [(0, 1, 2), (3, 4, 5)]          # candidate hyperedges
NOISE = 0.6                                # observation noise


def make_sample(active: int):
    """Time series for 6 nodes; `active` selects which triple is XOR-coupled."""
    X = rng.choice([-1.0, 1.0], size=(N_NODES, T))       # independent signs
    i, j, k = TRIPLES[active]
    X[k] = X[i] * X[j]                                   # pure 3-way synergy
    return X + NOISE * rng.standard_normal((N_NODES, T))


def dataset(n):
    y = rng.integers(0, 2, size=n)
    X = np.stack([make_sample(a) for a in y])
    return X, y


# ---------------------------------------------------------------- features --
def pairwise_features(X):
    """Graph model: one round of correlation-weighted message passing, then
    node-level second-moment readout + all pairwise correlations."""
    feats = []
    for x in X:
        C = np.corrcoef(x)
        np.fill_diagonal(C, 0.0)
        deg = np.abs(C).sum(1) + 1e-9
        A = C / deg[:, None]                 # normalised adjacency
        H = A @ x                            # message passing on signals
        iu = np.triu_indices(N_NODES, 1)
        feats.append(np.concatenate([(H * x).mean(1), C[iu]]))
    return np.array(feats)


def hyperedge_features(X):
    """Hypergraph model: aggregate each candidate hyperedge with a triple
    product -- the natural third-order statistic E[x_i x_j x_k]."""
    return np.array([[x[list(tr)].prod(0).mean() for tr in TRIPLES] for x in X])


# ------------------------------------------------------- tiny trained head --
def train_logistic(F, y, steps=4000, lr=0.5):
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    w, b = np.zeros(F.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(F @ w + b)))
        g = p - y
        w -= lr * (F.T @ g / len(y) + 1e-3 * w)
        b -= lr * g.mean()
    return w, b, F.mean(0), F.std(0)


def accuracy(model, Xte, yte, feat_fn):
    w, b, mu, sd = model
    F = (feat_fn(Xte) - mu) / (sd + 1e-9)
    return ((F @ w + b > 0).astype(int) == yte).mean()


if __name__ == "__main__":
    Xtr, ytr = dataset(400)
    Xte, yte = dataset(400)

    acc = {}
    for name, fn in [("Graph model (pairwise)", pairwise_features),
                     ("Hypergraph model (triadic)", hyperedge_features)]:
        model = train_logistic(fn(Xtr), ytr)
        acc[name] = accuracy(model, Xte, yte, fn)
        print(f"{name:30s} test accuracy: {acc[name]:.3f}")

    # ---- figure: mean |pairwise corr| vs triple statistic, and accuracies --
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x0 = make_sample(0)
    C = np.corrcoef(x0); np.fill_diagonal(C, 0)
    im = axes[0].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("Pairwise correlations\n(active triple invisible)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    tstats = hyperedge_features(np.stack([make_sample(a) for a in [0]*50 + [1]*50]))
    axes[1].scatter(tstats[:50, 0], tstats[:50, 1], label="class 0", alpha=.6)
    axes[1].scatter(tstats[50:, 0], tstats[50:, 1], label="class 1", alpha=.6)
    axes[1].set_xlabel("triple stat e=(0,1,2)"); axes[1].set_ylabel("triple stat e=(3,4,5)")
    axes[1].set_title("Hyperedge statistics\n(classes separate cleanly)"); axes[1].legend()

    names = list(acc); vals = [acc[n] for n in names]
    axes[2].bar(["Graph\n(pairwise)", "Hypergraph\n(triadic)"], vals,
                color=["#888888", "#1F3864"])
    axes[2].axhline(0.5, ls="--", c="k", lw=1, label="chance")
    axes[2].set_ylim(0, 1.05); axes[2].set_title("Test accuracy"); axes[2].legend()
    for i, v in enumerate(vals):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout(); plt.savefig("results.png", dpi=150)
    print("wrote results.png")
