# GraphSAGE reference-model NPZ pilot for COMP8240

These scripts use the GraphSAGE code already present in `baselines/graphsage-simple/graphsage/` of the H2GCN repository. They do **not** replace or edit that code and do **not** touch the existing TensorFlow/H2GCN Docker image.

## Install

From the project root (`~/Documents/H2GCN`), extract the accompanying ZIP with paths preserved, then run:

```bash
docker build --platform linux/arm64 -t graphsage-repro:pt251 -f Dockerfile.graphsage .
bash scripts/run_graphsage_pilot.sh
```

The image uses native ARM64 on Apple Silicon with Python 3.10 and PyTorch 2.5.1. The pilot trains for 20 epochs on `sbm_h0.10_r1_seed2027.npz`. **This is a smoke test only, not a final comparable model score.** It creates `results/graphsage_sbm_h010_r1_h128_lr07_pilot.json` and `repro_logs/graphsage_sbm_h010_r1_h128_lr07_pilot.log`.

## Exact modifications from the original runner

- Reuses the original fork's two-layer GraphSAGE-mean classes, neighbour sample counts `[5, 5]`, SGD optimizer, and one minibatch of 256 per epoch.
- Loads graph CSR NPZ and the verified `shared_split_seed42.npz`, rather than relying on the broken `dataset.py` symlink or signac dataset workspace.
- Replaces empty neighbourhood sets of isolated nodes with a self-neighbour **only for those isolated nodes**; this is essential because the original mean aggregator may divide by zero for an empty set. The JSON reports the number affected.
- Uses explicit `seed=123`, stores the best model by validation accuracy (validation loss for ties), restores those weights, and evaluates the test set only once at the selected checkpoint.
- Uses PyTorch 2.5.1 on Python 3.10. The fork's unmodified experiment runner and original PyTorch versions were not executed.
- Keeps raw node features (dense representation of CSR features), without adding row normalization. Note that our independent GCN baseline uses row normalization, a preprocessing difference to disclose when comparing methods.
- Validation and test neighbour sampling remain stochastic as in the reference model; results are seed-specific.

## Final experiment plan

The original config has four settings: hidden `[128, 64]` and learning rate `[0.7, 0.1]`, each trained for 500 epochs. Across 30 synthetic graphs this is 120 runs. Select the best setting for each graph based on validation accuracy and validation loss, with no test-selection leakage. The full runner will follow after a successful reference-source pilot.
