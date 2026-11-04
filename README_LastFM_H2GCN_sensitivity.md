# LastFM-Asia H2GCN-2 validation-only sensitivity analysis

This package adds three scripts. **It does not overwrite** the existing H2GCN trainer, previous model results or checkpoints.

The previous LastFM three-seed comparison shows that the fixed H2GCN-2 setting is reproducibly behind GCN and MLP. This is an exploratory, predeclared sensitivity sweep using **only validation accuracy** to choose a configuration; test labels are never scored by candidate training. All settings retain two-hop H2GCN-2 and the original NPZ loader, row-normalised node features, fixed LastFM split seed42 and training seed123.

## Commands

```bash
cd ~/Documents/H2GCN
unzip -o ~/Downloads/LastFM_H2GCN_sensitivity.zip -d .
bash scripts/run_lastfm_h2gcn_sensitivity.sh dry-run
bash scripts/run_lastfm_h2gcn_sensitivity.sh pilot
bash scripts/run_lastfm_h2gcn_sensitivity.sh full
```

The pilot tests the highest-memory hidden128 setting for 20 epochs and is not included in the selection. Full runs use 2,000 maximum epochs and patience 150. All five new candidates save under `results/lastfm_h2gcn_sensitivity/{config}`. The reference is the preexisting `results/lastfm_h2gcn/full_seed123` (hidden64, lr0.01, dropout0.5, L2=0.0005).

The five new candidates change one parameter at a time: hidden size32, hidden size128, learning rate0.003, dropout0.2, or L2 penalty0.00005. The existing reference makes six compared settings. The summariser selects maximum **validation accuracy** with **minimum validation loss** as the tie-breaker, matching the prior checkpoint criterion. The summary includes validation macro-F1 as a diagnostic, but does not use it to select the winner. Any test evaluation of the selected configuration is a separate, subsequent step.

## Output

- `results/lastfm_h2gcn_sensitivity/validation_sensitivity_summary.csv`
- `results/lastfm_h2gcn_sensitivity/validation_selected_config.json`
- Per-candidate `results.json`, training trace and TF best checkpoint under each config directory.
- Per-run terminal logs under `repro_logs/lastfm_h2gcn_sensitivity`.

**Important:** This is exploratory sensitivity analysis rather than an unbiased estimate of tuned H2GCN generalisation: the same fixed validation split is used for choosing configurations. Do not use any test results from other candidate runs in selection; this workflow deliberately never evaluates candidate test labels. The reference's test outcome is already known from the preceding study. If a new candidate wins, evaluate only that selected checkpoint once, rather than testing all candidates.

If the selected configuration is `reference`, no additional test evaluation is needed. Report the previously completed three-seed fixed H2GCN baseline and the sensitivity analysis separately.
