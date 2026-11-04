# Final test evaluation of validation-selected LastFM H2GCN-2

Do not retrain or rerun the sensitivity search. The config is frozen: `l2small`, H2GCN-2, 64 hidden units, LR 0.01, dropout 0.5, L2 0.00005, training seed 123, checkpoint best epoch 404. Six candidates were ranked using validation accuracy and validation loss only.

From `~/Documents/H2GCN`, install files from the ZIP:

```bash
unzip -o ~/Downloads/LastFM_selected_checkpoint_final_test.zip -d .
```

Check selection, checkpoint and file presence without evaluating the test set:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/workspace -w /workspace h2gcn-repro:v1 \
  python scripts/evaluate_lastfm_h2gcn_selected_once.py --check-only
```

Then perform one final test evaluation:

```bash
docker run --rm --platform linux/amd64 -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v "$PWD":/workspace -w /workspace h2gcn-repro:v1 \
  python scripts/evaluate_lastfm_h2gcn_selected_once.py
```

Outputs are created under `results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/`:
- `selected_test_results.json` with accuracy, macro-F1, weighted-F1, balanced accuracy, class coverage, config, and SHA-256 provenance.
- `test_class_metrics.csv` with per-class support, recall, precision and F1.
- `test_node_predictions.csv` with test-node labels and predictions.

The evaluator loads the original NPZ H2GCN data and model, verifies model-variable checkpoint coverage and repeatable restoration, and insists on matching recorded validation accuracy, macro-F1 and validation loss BEFORE computing any test metrics. It never trains and never alters source checkpoints. It will refuse to overwrite an existing test output directory. Keep this test result fixed after you see it.

Caution: `selected_test_results.json` is evidence for one seed and one fixed split. All comparative uncertainty estimates are specific to training-seed variability and do not generalise to graph/split shift.
