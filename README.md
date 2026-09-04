# Topological Deep Learning Demos

Self-contained, NumPy-first demonstrations from my PhD research on **geometric and
topological deep learning for brain network dynamics** (City St George's, University
of London). Everything is implemented from the mathematics directly — no ML
frameworks — and every figure regenerates with a single command.

| Demo | Question it answers | Run |
|---|---|---|
| [`hypergraph_vs_gcn/`](hypergraph_vs_gcn/) | Why can't pairwise (graph) models see higher-order interactions? | `python run_experiment.py` |
| [`dirac_wrcf/`](dirac_wrcf/) | How do clique filtrations + the Dirac operator quantify triadic structure? | `python run_demo.py` |

## Headline results

**Demo 1** plants a purely triadic (XOR-type) interaction where every pairwise
correlation is zero in expectation. A pairwise/message-passing model scores
**~0.54** (chance); a hypergraph model aggregating over candidate triples scores
**1.00** on the same data and training procedure.

**Demo 2** implements the Weighted Rank Clique Filtration (WRCF), signed boundary
operators B1/B2, the block Dirac operator D (with a numerical check that
D² = diag of the Hodge Laplacians), and the **Dirac spectral gap γ** — the weakest
supported topological mode. Weakening a network's triangles while holding edge
count fixed collapses γ (0.936 → 0.769 at τ=0.5), an effect invisible to
edge-level summaries.

## Setup

```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib
```

## Background

These demos accompany my PhD proposal *Geometric and Topological Deep Learning for
Modeling Higher-Order Interactions in Brain Network Dynamics* and my talk *Triadic
Topological Brain Networks: Weighted Rank Clique Filtration & Dirac Spectral
Geometry*. Results shown are single-realisation synthetic experiments intended as
clear, minimal illustrations of the underlying mathematics; averaging across seeds
and application to fMRI data (Natural Scenes Dataset) are part of the wider research
programme.

## Author

Joshua Chaim Joshi — Senior Software Engineer / ML researcher, London.
LinkedIn: linkedin.com/in/joshuachaimjoshi

MIT License.
