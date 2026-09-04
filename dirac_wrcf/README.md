# WRCF + Dirac spectral geometry

Implements the pipeline from my talk *Triadic Topological Brain Networks*:

1. **WRCF** — rank edges by weight, f(e_k)=k/m, clique complex at scale τ with
   f(σ)=max of its edges' filtration values.
2. **Boundary operators** B1 (edges→nodes), B2 (triangles→edges), consistent
   orientation.
3. **Dirac operator** D coupling nodes, edges and triangles in one operator;
   the code asserts numerically that D² = diag(L0, L1, L2) (Hodge Laplacians).
4. **Dirac gap** γ = min nonzero |eigenvalue| — γ² is the smallest non-zero
   Hodge eigenvalue: the weakest supported topological mode.

**Experiment.** A community network rich in strong triangles ("control") vs the
same network with 60% of its triangles weakened by damping their weakest edge
("degraded") — same nodes, same edge count at the analysis scale. At τ=0.5 the gap collapses from 0.986 to 0.768, and `gamma_curve.png` tracks γ(τ) across the
filtration. Interpretation: edges are less redundantly supported by triangles —
a structural-vulnerability signal that edge-level metrics with matched density
cannot express.

Single-realisation illustration; seed-averaged statistics and application to
neuroimaging (HCP/NSD) belong to the wider PhD programme.
Numbers quoted are from a fixed-seed run; exact values may vary slightly across NumPy versions, while the qualitative gap collapse is stable.
