# Reproducing H2GCN under Homophily and Structural Distribution Shift

**COMP8240 | Applications of Data Science | Replication and extension study**

This repository studies how the two-hop graph neural network **H2GCN** behaves when a graph changes from heterophilous to homophilous and when its structural distribution changes. It reproduces a selected result from Zhu et al. (2020), evaluates a **30-graph controlled SBM/DC-SBM benchmark**, and tests generalisation on the **LastFM-Asia** social network. The study compares H2GCN with independently adapted GCN and GraphSAGE baselines, and with an MLP on the real-world graph.

This is **not** the official H2GCN project. The model implementation and original experiment framework are credited to [GemsLab/H2GCN](https://github.com/GemsLab/H2GCN), based on [Zhu et al. (2020), *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*](https://arxiv.org/abs/2006.11468). The original upstream README is retained as [`README_UPSTREAM_H2GCN.md`](README_UPSTREAM_H2GCN.md).

**Start here:** [Reproducibility protocol](REPRODUCIBILITY.md) · [Docker and environment setup](DOCKER.md) · [Dataset generation scripts](scripts/) · [Curated results](results/)

## Research questions

1. How closely can the published H2GCN result on released `syn-cora` graphs be reproduced when the original evaluation split is unavailable?
2. How do H2GCN, GCN and GraphSAGE respond to increasing edge homophily on graphs with different degree and connectivity profiles?
3. What explains observed differences between SBM and degree-corrected SBM, particularly for isolated and low-degree nodes?
4. How well do the tested models transfer to a social-network node-classification task, and how sensitive is H2GCN to regularisation?

## Experiment overview

| Part | Data and protocol | Purpose |
|---|---|---|
| Original-paper check | Three published `syn-cora` realisations at edge homophily near 0.10, aligned reconstructed class-balanced split | Approximate replication of the paper's H2GCN-2 result |
| Controlled extension | SBM and degree-corrected SBM, 5 targets × 3 graph realisations × 2 families = **30 graphs** | Compare homophily and structural distribution shifts |
| Synthetic baselines | H2GCN-2, validation-selected GCN, validation-selected GraphSAGE | Measure architectural differences on matched graph realisations |
| Degree audit | Node degrees 0, 1–2, 3–5, 6+ | Examine failures related to node connectivity |
| External dataset | SNAP LastFM-Asia; 18 geographic classes, one fixed stratified split | Test external-domain performance and class imbalance |

**Controlled graph data:** Each synthetic graph contains 1,490 nodes, 1,433 features and five classes. Node attributes, labels and split are held fixed across conditions, whereas edges are regenerated. The common split consists of 370 training, 370 validation and 750 test nodes. The five target edge-homophily levels are 0.10, 0.30, 0.50, 0.70 and 0.90. Targets are not assumed to equal the empirically observed homophily; see `results/structural_grid_summary.csv`.

### Approximate original-paper reproduction

| Measurement | H2GCN-2 test accuracy |
|---|---:|
| Zhu et al. (2020), reported `syn-cora` at h ≈ 0.10 | **76.87 ± 0.43%** |
| This study, released graphs and reconstructed 25/25/50 split | **72.84 ± 1.27%** |

Values are mean ± sample standard deviation over three graphs. The original paper's split assignments were not available; the replacement split was mapped across graph realisations using feature/label signatures. Therefore, the roughly 4.03-point gap **is not evidence of a model defect or an exact failure to reproduce the paper's evaluation protocol**.

### Controlled homophily and structure experiments

Mean test accuracy across three graph realisations per condition (%):

| Target h | Family | H2GCN-2 | GCN | GraphSAGE |
|---:|---|---:|---:|---:|
| 0.10 | SBM | 71.60 | 33.60 | 69.47 |
| 0.10 | DC-SBM | 71.42 | 39.24 | 66.71 |
| 0.30 | SBM | 71.69 | 49.38 | 69.11 |
| 0.30 | DC-SBM | 70.40 | 50.89 | 67.51 |
| 0.50 | SBM | 77.69 | 70.53 | 75.69 |
| 0.50 | DC-SBM | 77.78 | 69.96 | 74.98 |
| 0.70 | SBM | 90.13 | 88.18 | 86.18 |
| 0.70 | DC-SBM | 87.87 | 86.53 | 82.93 |
| 0.90 | SBM | 97.16 | 96.93 | 95.78 |
| 0.90 | DC-SBM | 93.96 | 94.84 | 94.09 |

The H2GCN configuration was held fixed on the synthetic grid. GCN and GraphSAGE were selected per graph using their validation sets, so hyperparameter-search budgets are **not identical**. Reported graph realisations, rather than individual nodes, are the relevant replicate units. Detailed runs and validation-selection candidates are in `results/`.

**Structural interpretation:** DC-SBM graphs also contain more isolated nodes and smaller largest connected components than SBM graphs. Results therefore represent a **combined structural shift**, not a controlled causal test of degree variance alone. Degree-bin standardisation in `results/degree_robustness_three_model_composition.csv` is descriptive rather than causal.

### LastFM-Asia external-domain evaluation

The [SNAP LastFM-Asia dataset](https://snap.stanford.edu/data/feather-lastfm-social.html) contains 7,624 users, 27,806 undirected edges, 7,842 binary artist features and 18 imbalanced geographic classes. The fixed stratified split (seed 42) contains 1,898 training, 1,898 validation and 3,828 test nodes. The results below vary **training seeds 123, 124 and 125**, not independent train/test splits.

| Evaluated configuration | Test accuracy | Test macro-F1 |
|---|---:|---:|
| **H2GCN-2, validation-selected L2 = 0.00005** | **85.17 ± 0.03%** | **62.34 ± 0.33%** |
| GCN, evaluated configuration | 82.95 ± 0.47% | 53.82 ± 2.54% |
| MLP, evaluated configuration | 68.59 ± 0.34% | 36.04 ± 0.58% |
| H2GCN-2, original L2 = 0.0005 | 55.71 ± 0.11% | 16.14 ± 0.11% |

The six-candidate H2GCN sensitivity analysis selected its regularisation setting **using validation accuracy only**, with validation loss as a tie-breaker. The 85.17% result is a three-seed mean for that previously selected setting. Its advantage over the evaluated GCN is **configuration-specific**, not a claim that H2GCN always outperforms GCN: tuning budgets were different and all runs used one graph/split. Two minority classes had zero recall in the selected seed-123 test evaluation, despite high overall accuracy.

## Repository map

```text
h2gcn/                        Original H2GCN implementation
baselines/                    Original baseline source trees
scripts/generate_*grid.py     SBM and DC-SBM generation
scripts/calibrate_dcsbm.py    Degree-corrected graph calibration
scripts/audit_structural_grid.py
scripts/gcn_npz_tf2.py        GCN adapter, TensorFlow 2.2
scripts/graphsage_npz_reference.py
scripts/train_lastfm_h2gcn_*.py
scripts/lastfm_gcn_mlp_validated.py
scripts/summarize_*.py        Aggregation and validation selections
results/                      Curated CSV and JSON evidence
REPRODUCIBILITY.md            Protocols, caveats and output provenance
DOCKER.md                     Environment setup and verification
Dockerfile.h2gcn              Clean-build-tested H2GCN environment recipe
Dockerfile.graphsage          GraphSAGE PyTorch build recipe
.dockerignore                 Excludes data/logs/checkpoints from Docker builds
```

The complete list of results and the differences among training, validation and test scripts are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The repository deliberately excludes most raw archives, constructed `.npz` graphs, TensorFlow checkpoints, per-node prediction matrices and experiment logs.

## Getting started

The H2GCN TensorFlow image was rebuilt from the published Dockerfile on 4 November 2026 using `linux/amd64` emulation on Apple Silicon. The build passed package and data-loader verification, and the **saved LastFM checkpoint reproduced its recorded validation metrics** in the new image. This is a checkpoint-level validation, **not a full rerun of all experiments**. The LastFM checkpoint and NPZ datasets are not distributed in the public Git repository.

```bash
git clone https://github.com/navraaf/h2gcn-replication-structural-shift.git
cd h2gcn-replication-structural-shift

docker build --no-cache --platform linux/amd64 \
  -f Dockerfile.h2gcn -t h2gcn-repro:clean-test .

docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace -w /workspace h2gcn-repro:clean-test \
  python scripts/verify_h2gcn_environment.py
```

The verified clean environment uses Python 3.8.20, TensorFlow 2.2.0, NumPy 1.18.5, SciPy 1.4.1, NetworkX 2.4 and scikit-learn 0.23.2. The earlier working research image (`h2gcn-repro:v1`) had a different package history and is preserved separately. **Do not overwrite it** when testing a clean build.

If the fixed validation data and selected model checkpoint are locally available, the stronger verification is:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace -w /workspace h2gcn-repro:clean-test \
  python scripts/verify_lastfm_checkpoint_validation_only.py
```

This command reproduces the frozen LastFM validation accuracy (85.4057%), macro-F1 (63.6393%) and validation loss (1.08940) from the existing checkpoint without test-set scoring or retraining. It cannot run on an empty checkout because the relevant local graph, split and checkpoint are excluded from Git.

GraphSAGE uses a separate PyTorch image (see [`Dockerfile.graphsage`](Dockerfile.graphsage) and [`DOCKER.md`](DOCKER.md)). For full experiment protocols, additional package constraints and dataset construction steps, consult [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

To regenerate experiments, first obtain the released source datasets, then consult the exact argument syntax in the tracked generation and runner scripts and the corresponding README files. In particular, `constructed_data/`, `existing_data/` and some split/checkpoint paths are local artifacts not included in the public commit. **This is not a one-command rerun from an empty checkout.** Readers can independently inspect the committed aggregate CSV/JSON evidence without training models.

## Reproducibility, limitations and ethics

- **Split fidelity:** The paper's original synthetic split IDs are unavailable; only approximate paper replication is claimed.
- **Structural confounding:** DC-SBM changes degree dispersion, isolation and connectivity together. No causal effect of degree heterogeneity alone is identified.
- **Unequal tuning:** The evaluated GCN, GraphSAGE and H2GCN configurations have different validation-search protocols.
- **Repeat-unit distinction:** Synthetic summaries aggregate three graphs; LastFM summaries aggregate three initialisation seeds on one split.
- **Class imbalance and privacy:** LastFM labels describe users' locations and artist interests. Results are aggregate research analyses, not individual location-inference recommendations. Minority-class recall is reported and should not be hidden by overall accuracy.
- **Software age:** TensorFlow 2.2 and Python 3.8 are legacy dependencies. Isolate the environment in Docker; do not use it as a production service.

## Attribution

- Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*. NeurIPS. [Paper](https://arxiv.org/abs/2006.11468) · [Official source](https://github.com/GemsLab/H2GCN).
- Rozemberczki, B., & Sarkar, R. (2020). *Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models*. CIKM. [SNAP dataset and citation](https://snap.stanford.edu/data/feather-lastfm-social.html).

This is an academic replication and extension. The original authors retain credit for their code and scientific contributions; all modifications, new datasets and evaluation protocols should be identified separately. Refer to the original repository for upstream licensing terms.
