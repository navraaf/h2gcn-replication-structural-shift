# LastFM-Asia: per-class audit and matched GCN / MLP baselines

Uses the existing processed LastFM graph (`N=7,624; 18 classes`) and the exact
`lastfm_asia_split_seed42.npz` (1,898 train / 1,898 val / 3,828 test).
Nothing overwrites the successful H2GCN run at
`results/lastfm_h2gcn/full_seed123/`.

## 1. Install

From `~/Documents/H2GCN`:

```sh
unzip -o ~/Downloads/LastFM_followup_class_audit_and_baselines.zip -d .
```

## 2. Audit saved H2GCN test predictions (does not train anything)

```sh
bash scripts/run_lastfm_followup.sh audit
```

Outputs: `results/lastfm_class_audit/class_audit_summary.json`,
`class_performance_audit.csv`, and `confusion_matrix.csv`.
The audit independently recalculates test accuracy, macro-F1, and each class's
precision/recall/F1 from the saved per-node H2GCN predictions, checking them
against the earlier metrics. Report how many of 18 classes were never predicted,
which classes had zero recall, and whether the model concentrates predictions.

## 3. Pilot two external baselines (do not include pilot scores in report)

```sh
bash scripts/run_lastfm_followup.sh pilot
```

Runs GCN and a graph-free MLP (max 20 epochs, patience 12) to check the full
pipeline. The GCN is an independent TF2-compatible conventional two-layer
implementation, **not the unmodified TensorFlow 1.x implementation used by
Zhu et al.** The MLP has the *same* two-layer architecture but uses identity
adjacency instead of graph propagation. Both use hidden=64, dropout=.5,
Adam lr=.01, L2 coefficient=.0005, TF2.2 image `h2gcn-repro:v1`.

## 4. Full matched GCN and MLP evaluations

After the pilot succeeds:

```sh
bash scripts/run_lastfm_followup.sh baselines
```

Each model uses a 2,000-epoch cap and patience 150, with validation accuracy
and val-loss tie-break to select the checkpoint; test is evaluated only after
selection, exactly once. Output directories:

- `results/lastfm_baselines/full_gcn_seed123/`
- `results/lastfm_baselines/full_mlp_seed123/`

Each includes `results.json`, `test_node_predictions.csv`,
`test_class_metrics.csv`, and `train_validation_trace.csv`. These configurations
are **fixed**, not a LastFM-specific hyperparameter sweep. This limits
hyperparameter-selection bias but does not establish that either baseline is
optimally tuned. No test scores may be used to choose hyperparameters.

## 5. Repeat H2GCN seeds on the same fixed split

```sh
bash scripts/run_lastfm_followup.sh h2gcn-seeds
```

Runs the already-installed `train_lastfm_h2gcn_validated.py` at seeds 124 and
125. Uses split 42 for all runs. Outputs do not overwrite seed 123.
These 3 training seeds quantify sensitivity to optimization randomness, **not**
variation across independently sampled training/test splits.

## 6. Summarize available full runs

```sh
python3 scripts/summarize_lastfm_models.py
```

Output: `results/lastfm_comparison/lastfm_model_runs.csv` and
`lastfm_model_summary.csv`. Sample SD is omitted for models with one seed.

## Scientific interpretation

The original full H2GCN seed-123 run reached 55.80% test accuracy and 16.24%
macro-F1, against a 20.53% training-majority-label baseline, on fixed split42.
The difference between accuracy and macro-F1 suggests class imbalance or
class-specific failure and motivates a per-class diagnostic. Its origin cannot
be established from aggregate scores alone. Compare H2GCN, GCN and graph-free
MLP on exactly the same graph, features and split before making a real-world
robustness claim. GraphSAGE can be added in a separate memory-bounded evaluation.

All scripts are generated as standalone evaluation harnesses. Syntax and
non-TensorFlow audit logic were checked in the authoring environment. Actual
TensorFlow training must pass the Mac Docker pilot before results are used.
