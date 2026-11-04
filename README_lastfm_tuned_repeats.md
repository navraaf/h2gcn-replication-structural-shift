# LastFM-Asia: fixed validation-selected H2GCN repeat-seed evaluation

This package does **not** retune the model and does not overwrite the tested seed-123 checkpoint. Its configuration was selected from six candidates using validation data: hidden 64, Adam learning rate 0.01, dropout 0.5, and L2 0.00005.

The independent repeats use training seeds 124 and 125, fixed graph and split seed 42, a maximum of 2000 epochs, and patience 150. Each run selects its checkpoint on validation accuracy (validation loss breaks ties), verifies that all H2GCN model variables were restored exactly from the selected epoch, then evaluates the test set once. These are repeated initialisations, **not independent graph splits**.

## Install

From your repo root:

```bash
unzip -o ~/Downloads/LastFM_H2GCN_tuned_repeats.zip -d .
```

## Execute

```bash
bash scripts/run_lastfm_h2gcn_tuned_repeats.sh dry-run
bash scripts/run_lastfm_h2gcn_tuned_repeats.sh run
```

The runner checks the frozen seed-123 result at `results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/selected_test_results.json`. If it is missing or a run directory is partially populated, it will stop and not overwrite anything.

Results: `results/lastfm_h2gcn_tuned_repeats/tuned_lastfm_h2gcn_runs.csv` and `results/lastfm_h2gcn_tuned_repeats/tuned_lastfm_h2gcn_summary.csv`. Upload both for independent audit.

This training driver is an adaptation of the original H2GCN implementation and loader. It stores native TensorFlow checkpoint files, tolerates only the TensorFlow `save_counter` bookkeeping mismatch, and independently validates the restored model's full variables and validation accuracy before scoring test data.
