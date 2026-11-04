# LastFM-Asia: validation-selected H2GCN-2 evaluation

This package adds two new scripts under `scripts/`. It does not replace the existing `preprocess_lastfm.py`, NPZ data or original H2GCN files.

## First commands

From the repository root, after unzipping:

```bash
bash scripts/run_lastfm_h2gcn.sh check
bash scripts/run_lastfm_h2gcn.sh pilot
```

The check validates the graph and partition without running TensorFlow training. The pilot trains for up to 20 epochs and produces complete output files to test the full data-to-metrics pipeline. Its accuracy is **not** the final benchmark.

After the pilot passes, use:

```bash
bash scripts/run_lastfm_h2gcn.sh full
```

This trains for up to 2,000 epochs with patience 150 and outputs metrics under `results/lastfm_h2gcn/full_seed123`. It keeps the previous five-epoch smoke test and synthetic checkpoints untouched. Validation accuracy (with validation loss tie-break) chooses the checkpoint. Test labels are evaluated only once, after best-weight restoration.

## Output

- `results.json`: selected epoch, accuracy, macro-F1, weighted-F1, balanced accuracy, majority-class comparison, dataset and configuration metadata.
- `test_node_predictions.csv`: one line per test node, with true and predicted labels.
- `test_class_metrics.csv`: per-class precision/recall/F1/support.
- `class_support_by_split.csv`: class counts across the three partitions.
- `train_validation_trace.csv`: learning curve and selection evidence.
- `best_checkpoint/ckpt-*`: TensorFlow weights of validation-selected model.

The script reuses the original H2GCN architecture, its `NPZData` loader and preprocessing, but uses a **custom evaluation driver** to prevent repeated test-set inspection during training. The checkpoint rule (accuracy with loss tie-break) is slightly different from the reference driver's latest-tie-wins rule. Clearly document that distinction in the report. Do not compare this new run directly with previously completed synthetic runs as a controlled graph-only shift: LastFM also changes domain, label classes, features, scale and splits.

## Cautions

- The earlier LastFM five-epoch test (31.30% test accuracy) was only a smoke test, not a benchmark.
- One fixed split and one training seed are preliminary. Replicate across multiple predefined seeds or splits, and run relevant baselines, before reporting a stable real-world result.
- There are 18 imbalanced classes. Macro-F1 is essential; accuracy alone is insufficient.
- LastFM's user-artist features include zero rows. The original loader can produce divide-by-zero warnings for these rows. Such warnings were present in the earlier smoke test, but a nonfinite metric is an error and stops the new runner.
- H2GCN's two-hop propagation on this larger graph may require substantial memory. The pilot is meant to catch this before a full run.
- The full runner is not resumable mid-training. It refuses to overwrite an existing partial or completed run. If interrupted, preserve the partial directory for debugging rather than deleting valuable outputs.
