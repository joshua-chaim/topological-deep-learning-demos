# Hypergraph vs Graph: seeing higher-order interactions

**Claim.** Interactions that exist only at third order are provably invisible to
pairwise statistics. This demo makes that concrete.

**Construction.** Six nodes emit random ±1 signals plus Gaussian noise. In each
sample, one candidate triple — (0,1,2) or (3,4,5) — is XOR-coupled: the third
node's signal is the product of the other two. Because E[x_i x_j] = 0 for every
pair, the correlation matrix carries no class information (left panel of
`results.png`). The third-order statistic E[x_i x_j x_k] over the coupled triple
equals 1, over the uncoupled triple 0 (middle panel).

**Models.** Identical logistic heads trained by gradient descent; only the
features differ. The graph model gets one round of correlation-weighted message
passing plus all pairwise correlations; the hypergraph model gets one triple-
product statistic per candidate hyperedge.

**Result.** Graph ~0.54 (chance), hypergraph 1.00. The gap is structural, not a
tuning artefact — no amount of training rescues features that carry no signal.

Relation to the wider research: this is the minimal version of the argument for
lifting brain functional-connectivity graphs to hypergraphs/simplicial complexes
before learning.
