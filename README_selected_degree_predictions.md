# Selected GCN and GraphSAGE node-prediction export

## Purpose
The user already completed **270 GCN hyperparameter candidates** and **120 GraphSAGE candidates**. This package **does not repeat those sweeps**. It reruns only the single validation-selected configuration for each of 30 graphs for both baseline models, and captures test-node predictions from the selected checkpoint. H2GCN's 30 checkpoints have already been restored independently. The final join verifies accuracy, node IDs, labels and original graph degrees before producing degree-binned comparison tables.

### Scientific safeguards
- The GCN export script is a copy of the previously used TF2-compatible GCN, with only CSV output code added after restoration of the best validation checkpoint. The GCN is **not the original paper's TF1.x code**.
- The GraphSAGE export script reuses the authors' GraphSAGE fork encoders/aggregators, and captures predictions from its **existing fixed-seed test-sampling evaluation**. The prior NPZ adaptation's isolated-node self-neighbour fallback is retained. It is **not an unchanged reference run**.
- Test labels are evaluated after validation-based checkpoint selection. Existing hyperparameter candidate JSONs are not modified.
- Each rerun must reproduce its original *validation* and *test* accuracy to within 0.05 percentage points, and GraphSAGE must reproduce the selected checkpoint epoch. Otherwise the process stops.
- CSVs are cross-validated on each node's ID, class label and true graph degree before calculating comparisons.
- Final 3-replicate SDs should be interpreted across graph realisations, not across the 2,250 pooled test observations as independent samples.
- Reweighting the degree-bin mix is a **descriptive standardisation**, not causal identification.

## Install
From the H2GCN repository root:

```bash
cd ~/Documents/H2GCN
unzip -o ~/Downloads/Selected_baseline_degree_predictions.zip -d .
```

Do not overwrite `scripts/gcn_npz_tf2.py` or `scripts/graphsage_npz_reference.py`. This ZIP uses distinct script names.

## Prerequisites
- Existing Docker images `h2gcn-repro:v1` and `graphsage-repro:pt251`.
- Original grid graphs and `constructed_data/shared_split_seed42.npz`.
- `results/gcn_selected_by_validation.csv` (30 rows).
- `results/graphsage_selected_by_validation.csv` (30 rows).
- `results/h2gcn_node_predictions/node_predictions.csv` and `checkpoint_accuracy_verification.csv` from the prior H2GCN extraction.
- Baselines and their experimental scripts already installed under the normal repository root.

## Pilot, then full 60 selected reruns

```bash
python3 scripts/run_selected_degree_predictions.py --model all --limit-per-model 1
```

The pilot runs one GCN and one GraphSAGE. Both must print `PASS` and match the earlier validation-selected results; a mismatch halts execution and must be investigated. After this succeeds:

```bash
python3 scripts/run_selected_degree_predictions.py --model all
```

The runner checks existing outputs and resumes safely from the first missing or invalid result. It writes `results/gcn_node_predictions/<graph>.csv`, `results/graphsage_node_predictions/<graph>.csv` and separate JSON/logs. It does not change any original candidate result JSON.

## Summarise node-level robustness results

```bash
python3 scripts/summarize_three_model_degree_robustness.py
```

The summariser checks all 90 graph-model combinations and emits:

1. `results/degree_robustness_accuracy_checks.csv` (90 verification records).
2. `results/degree_robustness_three_model_runs.csv` (360 run-by-degree records).
3. `results/degree_robustness_three_model_summary.csv` (120 mean, SD and pooled accuracy entries).
4. `results/degree_robustness_three_model_composition.csv` (15 descriptive degree-bin standardisations).
5. `results/degree_robustness_three_model_paired_nodes.csv` (22,500 node-matched test observations).

**Stop and share the output if either pilot run fails or reports a score mismatch.** Do not use mismatched predictions in the paper.
