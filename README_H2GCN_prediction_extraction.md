# H2GCN checkpoint-based node-level prediction extraction

## Purpose

Reconstructs the original H2GCN-2 model using the local NPZ loader, restores the **best-validation** TensorFlow checkpoint for each graph, and extracts predicted labels and node degrees. Both **test and validation accuracy are required to match the original 30-row H2GCN experiment CSV** within 0.01 percentage points (to accommodate two-decimal rounding).

The script does not train or modify any checkpoint; it uses `tf.train.Checkpoint(model=model)` and the original H2GCN preprocessing. It deliberately **does not invoke** `modules/logger.py:init_checkpoint_path`, which **moves existing checkpoint folders** when called.

## Install in H2GCN repository

From Terminal:

```bash
cd ~/Documents/H2GCN
unzip -o ~/Downloads/H2GCN_node_prediction_extractor.zip -d .
```

## Pilot (one saved model)

```bash
bash scripts/run_h2gcn_prediction_extraction.sh pilot
```

The pilot is the first sorted manifest row, DC-SBM at h=0.10, replicate 1. The output must show matching validation and test accuracy and end with `PASS` before you run the full extraction.

If it fails, **stop and share the log** `repro_logs/h2gcn_checkpoint_extraction_pilot.log`. Do not modify checkpoint files or retrain.

## Complete extraction

```bash
bash scripts/run_h2gcn_prediction_extraction.sh full
```

Expected final message: `PASS: 30 checkpoints restored; all recovered test accuracies matched.`

## Results

Saved under `results/h2gcn_node_predictions/`:

- `checkpoint_accuracy_verification.csv`: all 30 validation/test parity comparisons.
- `degree_binned_test_accuracy.csv`: 120 graph-by-degree-bin records, with counts and accuracy.
- `node_predictions.csv`: per-node predicted and true class, graph ID, split, degree, correctness (44,700 rows).
- `dcsbm_h0.10_r1_predictions.npz` etc.: node logits, labels, degree, and split indices (30 files).

The program writes progress after each graph. If restoration fails mid-grid, previously verified results remain in the output directory, but the `PASS: 30` message will be absent; partial results should **not** be treated as complete. Re-running `full` recomputes the predictions from checkpoints, without retraining.

## Methodological safeguards

- The split file is always `constructed_data/shared_split_seed42.npz`.
- The model specification is `M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO`, with normalized adjacency hop radii 1 and 2.
- Training-style preprocessing consists of row-normalized features followed by adjacency diagonal removal and `getAdjNormHops` preprocessing.
- Only model weights (not optimizer state) are needed for inference. The restoration status verifies existing model variables were matched.
- The original logits are passed to argmax to obtain predicted classes.
- Degrees exclude diagonal edges and are binned as `0`, `1-2`, `3-5`, `6+`.
- Results apply to these **synthetic** SBM and DC-SBM graphs, not to LastFM-Asia.

## Known constraint

The source ZIP supplied for development did not include all model dependency modules or the large checkpoints, so the extractor was **syntax and manifest-logic tested here**, but full TensorFlow restoration must be confirmed by the pilot on your Mac. The source classes and loaders are loaded from the *actual* H2GCN repository during execution.
