# Full GraphSAGE benchmark | COMP8240

## Execution

Extract `GraphSAGE_full_grid_scripts.zip` inside the existing repository, after the successful pilot:

```bash
cd ~/Documents/H2GCN
unzip -o ~/Downloads/GraphSAGE_full_grid_scripts.zip -d .
bash scripts/run_graphsage_full_grid.sh
python3 scripts/summarize_graphsage_grid.py
```

This performs 120 reference-architecture GraphSAGE training runs (30 graphs x four configurations). Each run uses 500 epochs, SGD, batch size 256, two layers with [5,5] neighbor sampling, and seed 123. The two hidden configurations are 64 and 128, learning rates 0.1 and 0.7. Valid results are reused on restart; if a run fails, it stops and can be rerun.

The archive **updates** `scripts/graphsage_npz_reference.py` to version 2, which makes validation/test neighbour sampling repeatable while preserving the training RNG state. This is important because the reference Encoder otherwise samples neighbours randomly during every validation evaluation.

Before starting the full grid, rerun the 20-epoch pilot once using `bash scripts/run_graphsage_pilot.sh` to verify compatibility. The pilot result will now represent the updated adapter; do not interpret it as a final research score.

Because the full grid requires the new sampling metadata, the sweep script will refuse to reuse any old-format full-run JSONs.

## Output

- `results/graphsage_tuning/*.json`: 120 JSON files
- `repro_logs/graphsage_tuning/*.log`: training logs
- `results/graphsage_all_validation_candidates.csv`
- `results/graphsage_selected_by_validation.csv`
- `results/h2gcn_vs_graphsage_validation_selected.csv`
- `results/h2gcn_gcn_graphsage_paired_runs.csv` when existing GCN selected CSV is present

The summarizer requires `results/h2gcn_grid_runs.csv` in the repository and expects the same shared split used in the earlier experiments. The final comparison is paired by family, target homophily, and graph realisation. Test outcomes are not used for tuning; validation accuracy and loss decide selection.

## Scientific cautions

- This is an adaptation reusing original GraphSAGE components, not an unchanged run of the original authors' entire experiment runner.
- Zero-degree nodes use a self-neighbour fallback, a topology intervention (counted in each JSON).
- Neighbourhood sampling at validation and testing is seeded *deterministically per evaluation* rather than resampled every epoch. This controls a source of checkpoint-selection noise, while preserving stochastic training (the evaluation RNG states are restored). Single-sample neighbour evaluation still estimates, rather than exactly marginalises, sampling uncertainty. Treat standard deviations across three graphs as graph-realisation variation, not independent training-seed variation.
- Original GraphSAGE ran one minibatch per epoch, and this adapter retains that behaviour. Different models may have different compute budgets and preprocessing conventions.
- Parameter sweep is exactly four settings as shown in the original H2GCN synthetic GraphSAGE configuration. Report values faithfully, including underperformance and unexpected results.
