# Reproducibility and experimental protocol

## Project
**Reproducing H2GCN and Stress-Testing Robustness under Controlled Homophily and Structural Distribution Shift**

This is a replication and extension of Zhu et al. (2020), *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs* (NeurIPS 2020), using the authors' [H2GCN implementation](https://github.com/GemsLab/H2GCN). This file documents this project's data, experimental comparisons, validation rules, and limitations. `README.md` retains the upstream project description.

## Environments and reproducibility status

- The original H2GCN experiments use Python 3.8, TensorFlow 2.2.0, NumPy 1.18.5, SciPy 1.5.0, NetworkX 2.4 and the locally built `h2gcn-repro:v1` Docker image, run as Linux `amd64` on an Apple Silicon Mac.
- The adapted GraphSAGE baseline uses PyTorch in a separate local image, `graphsage-repro:pt251`, with `Dockerfile.graphsage`.
- **Important limitation:** The portable Docker recipe that reconstructs the full `h2gcn-repro:v1` image from scratch is not yet verified as part of this release. Local image tags alone are insufficient for another researcher. Add a pinned build recipe and test a clean build before claiming full reproduction from a clean checkout.
- Model checkpoints, large dataset archives, and most individual training logs are kept locally rather than committed. Compact summary data are in `results/`.
- Paths and runners may assume the `h2gcn-repro:v1` / `graphsage-repro:pt251` images and preprocessed graph files already exist. These are **documented study artifacts, not yet a one-command clean-room rerun**.

## Paper replication and split caveat

The released `syn-cora` h≈0.1 graphs were evaluated with H2GCN-2 under a deterministic reconstructed class-balanced train/validation/test split (25%/25%/50%, seed 42), because the original paper's exact synthetic split identifiers were unavailable. Three released realisations gave **72.84 ± 1.27%** test accuracy, compared with **76.87 ± 0.43%** reported in the original paper. This comparison is approximate, **not an exact reproduction of the published evaluation protocol**. See `repro_logs/REPRODUCTION_NOTES.txt` if available.

## Constructed structural-shift dataset

- Two graph families: stochastic block model (**SBM**) and degree-corrected SBM (**DC-SBM**).
- Five target edge-homophily levels: **0.10, 0.30, 0.50, 0.70, 0.90**; three independently constructed graph realisations per family and level (**30 graphs** total).
- All graphs contain **1,490 nodes**, **1,433 Cora-derived features** and **five classes**. Graph conditions share the node features, labels and 25/25/50 split; the graph edges vary.
- Generators: `scripts/generate_sbm_grid.py`, `scripts/generate_dcsbm_grid.py`, and `scripts/calibrate_dcsbm.py`. Their local output graph NPZ files are under `constructed_data/sbm_grid/` and `constructed_data/dcsbm_grid/`; the split file is `constructed_data/shared_split_seed42.npz`.
- Inspect generation scripts for the precise CLI and source feature/label input files. Required original feature/label inputs are **not bundled** in this compact release.
- Validation: `scripts/audit_structural_grid.py` produced `results/structural_grid_runs.csv` (30 rows) and `results/structural_grid_summary.csv` (10 conditions). Edges are undirected and loop-free; actual homophily should be reported rather than assuming the target was attained exactly.

**Major confound:** SBM and DC-SBM differ not only in degree dispersion but also in the prevalence of isolated nodes and largest-connected-component size. Structural effects should not be interpreted as the causal impact of degree heterogeneity alone.

## Synthetic model comparison

- **H2GCN-2:** fixed architecture `M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO`, 2-hop aggregation, 2,000-epoch maximum. See H2GCN grid runners, result summaries and `results/h2gcn_grid_runs.csv`.
- **GCN:** independent TensorFlow 2.2 adaptation; validation sweep over hidden dimensions **16, 32, 64** and early-stopping patience **40, 100, 200**, selecting per graph on validation data (270 candidates). Source `scripts/gcn_npz_tf2.py`, runner `scripts/run_gcn_validation_sweep.sh`.
- **GraphSAGE:** adapted original GraphSAGE modules with a shared NPZ loader; hidden dimensions **64, 128** and learning rates **0.1, 0.7**; validation selection per graph (120 candidates). Source `scripts/graphsage_npz_reference.py`, runner `scripts/run_graphsage_full_grid.sh`. The adapter uses an explicit isolated-node fallback and deterministic evaluation sampling.
- **MLP:** graph-free baseline on the common features and split; three training seeds. Unlike the graph families, its three repetitions vary training initialisation, not graph realisations.
- Outputs: `results/h2gcn_grid_summary.csv`, `results/gcn_selected_by_validation.csv`, `results/graphsage_selected_by_validation.csv`, and `results/h2gcn_gcn_graphsage_paired_runs.csv`.
- **Comparability limitation:** H2GCN is fixed-configuration; GCN and GraphSAGE are validation-tuned per graph. All three share graph and split for each graph realisation, but tuning protocols are not identical.

## Degree-specific error analysis

The trained synthetic H2GCN checkpoints were restored; GCN and GraphSAGE validation-selected configurations were rerun to export per-node predictions. Accuracy was reconciled against the previously saved aggregate results before analysing degrees **0**, **1–2**, **3–5**, and **6+**. Compact outputs include:

- `results/degree_robustness_accuracy_checks.csv`
- `results/degree_robustness_three_model_runs.csv`
- `results/degree_robustness_three_model_summary.csv`
- `results/degree_robustness_three_model_composition.csv`

Comparisons of the original and degree-standardised accuracy gaps are **descriptive decompositions**, not causal attributions. The 750 test nodes per graph are not independent graph realisations.

## LastFM-Asia external-domain evaluation

LastFM-Asia comes from Stanford SNAP. Preprocessing is in `scripts/preprocess_lastfm.py` and produces `existing_data/lastfm_asia/processed/lastfm_asia_h2gcn.npz` and `lastfm_asia_split_seed42.npz`. Observed dataset: **7,624 nodes**, **27,806 undirected edges**, **7,842 binary features**, **18 imbalanced classes**, zero-feature nodes **173**, and split **1,898 / 1,898 / 3,828** (train/val/test). It has no isolated nodes. Source raw files and processed NPZ are not committed in the compact release.

All models evaluate the same fixed split and report test accuracy, macro-F1, weighted-F1, and balanced accuracy. Three training seeds (123–125) were run for the tested baseline configurations.

| Model | Test accuracy (mean ± sample SD) | Macro-F1 (mean ± sample SD) |
|---|---:|---:|
| H2GCN, validation-selected L2=0.00005 | 85.17 ± 0.03% | 62.34 ± 0.33% |
| GCN, tested configuration | 82.95 ± 0.47% | 53.82 ± 2.54% |
| MLP, tested configuration | 68.59 ± 0.34% | 36.04 ± 0.58% |
| H2GCN, original L2=0.0005 | 55.71 ± 0.11% | 16.14 ± 0.11% |

H2GCN sensitivity used **six candidates**, selected by maximum validation accuracy with validation loss as tie-break; the `L2=0.00005` model selected at **epoch 404** was evaluated on the fixed test split, then repeated with seeds 124 and 125. See `results/lastfm_h2gcn_sensitivity/validation_sensitivity_summary.csv`, `validation_selected_config.json`, selected test JSON, and tuned-repeat summaries. The repeated runs reuse the selected hyperparameters and **same fixed split**.

**Limitations:** H2GCN and GCN tuning budgets differ, GCN sometimes reaches the 2,000-epoch cap, the fixed test split was used to examine the earlier reference model before the final tuned H2GCN evaluation, and some minority geographic classes remain poorly predicted. This is an external-domain evaluation, **not an isolated structural-shift causal intervention**.

## How to inspect and reproduce

1. Obtain the original H2GCN repository and the same released `syn-cora` feature/label data; reconstruct the fixed class-balanced split and generated SBM/DC-SBM graphs using the tracked generation scripts. Validate with `scripts/audit_structural_grid.py`.
2. Build/pin the two Docker environments as above. Run the documented grid and baseline scripts. **Before claiming clean reproducibility, independently test a fresh environment and check script arguments.**
3. Recompute validation selection from the candidate CSVs and compare the paired summaries. Inspect `scripts/summarize_*` for the selection rules.
4. Obtain LastFM-Asia from Stanford SNAP; preprocess it with `scripts/preprocess_lastfm.py`. Run the H2GCN, GCN/MLP and sensitivity workflows documented in the respective `README_LastFM_*.md` files.
5. The summary CSVs are committed evidence; full checkpoints or graphs are not required to **read and validate the reported aggregate results**, but are required to run the prediction-recovery analyses from scratch.

## Academic attribution and ethics

Credit the original H2GCN paper and implementation; identify baseline adaptations separately. The LastFM geographic-label prediction task has uneven class support and potential privacy/fairness implications, so class-wise recall and macro-F1 should accompany accuracy. State all dataset provenance, constraints and deviations from the original paper's reported splits and environment.
