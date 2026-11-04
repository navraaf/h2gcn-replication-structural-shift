# Docker environments and verification

This project used two model environments: a legacy TensorFlow 2.2.0 environment for H2GCN and the adapted GCN, and a separate PyTorch environment for GraphSAGE. On **4 November 2026**, `Dockerfile.h2gcn` was successfully rebuilt from scratch on Apple Silicon using Docker's `linux/amd64` emulation. The clean image passed package and NPZ loader checks, then restored the previously selected LastFM H2GCN checkpoint and reproduced its recorded **validation** accuracy, macro-F1 and loss. No retraining, test-set scoring or checkpoint changes occurred during this verification. This is not a full independent rerun of the 30-graph benchmark or the three-seed LastFM experiments.

The public repository does not include the graph NPZ archives and trained TensorFlow checkpoints needed for the stronger checkpoint verification. Environment-only checks can be run from a fresh checkout; checkpoint verification requires the locally produced data and checkpoint.

## 1. Platform and known working environments

| Component | Platform | Language/runtime | Key dependencies | Image name |
|---|---|---|---|---|
| H2GCN-2 and TF GCN | Linux `amd64` | Python 3.8 | TensorFlow 2.2.0, NumPy 1.18.5, SciPy 1.5.0, NetworkX 2.4 | `h2gcn-repro:v1` |
| H2GCN and GCN, clean rebuild | Linux `amd64` | Python 3.8.20 | TensorFlow 2.2.0, NumPy 1.18.5, SciPy 1.4.1, NetworkX 2.4 | `h2gcn-repro:clean-test` |
| Adapted GraphSAGE | Linux `arm64` on Apple Silicon | Python 3.10 | PyTorch 2.5.1, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2 | `graphsage-repro:pt251` |

The original experiments used `h2gcn-repro:v1`; the independently rebuilt image is `h2gcn-repro:clean-test` with SciPy 1.4.1. The original working image reported SciPy 1.5.0. This version difference means the Docker recipe is a verified **compatible reconstruction**, not a bit-for-bit clone of the original runtime. An image tag is not a portable content hash.

### Why `amd64` for TensorFlow 2.2?

TensorFlow 2.2.0 supplies a CPython 3.8 Linux **x86-64** wheel, but not a corresponding historic Apple-Silicon Python 3.8 wheel. On an Apple Silicon Mac, Docker Desktop can execute the Linux `amd64` container through CPU emulation. Expect slower training than native CPU; make sure Docker Desktop can emulate the architecture.

## 2. Capture provenance from the existing working H2GCN image

These commands capture the old working environment for comparison. Preserve its original tag; do not overwrite it with the clean image.

```bash
cd ~/Documents/H2GCN
mkdir -p docker/provenance

docker image inspect h2gcn-repro:v1 --format '{{.Id}}' > docker/provenance/working_image_id.txt
docker image inspect h2gcn-repro:v1 --format '{{json .Config}}' > docker/provenance/working_image_config.json
docker history --no-trunc h2gcn-repro:v1 > docker/provenance/working_image_history.txt

docker run --rm --platform linux/amd64 h2gcn-repro:v1 \
  python -m pip freeze > docker/provenance/working_pip_freeze.txt
```

Review these provenance files before publishing them: environment variables or image history can contain sensitive tokens, service endpoints or local paths. **Do not commit the raw image config/history without manual review.** The safe alternative is to publish only a manually reviewed dependency list and image hash.

## 3. Build a new H2GCN image without overwriting the old one

```bash
cd ~/Documents/H2GCN

docker build --no-cache --platform linux/amd64 \
  -f Dockerfile.h2gcn -t h2gcn-repro:clean-test .
```

This exact clean-build command **passed on 4 November 2026**. It uses Python 3.8.20 for Linux `amd64` and pins SciPy 1.4.1 to satisfy TensorFlow 2.2.0 dependency resolution. The distinct `clean-test` tag preserves `h2gcn-repro:v1` and all existing checkpoints.

### Verify the installed packages and the H2GCN loader

```bash
cd ~/Documents/H2GCN

docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace -w /workspace h2gcn-repro:clean-test \
  python scripts/verify_h2gcn_environment.py
```

This check **passed on 4 November 2026** with Python 3.8.20, TensorFlow 2.2.0, NumPy 1.18.5, SciPy 1.4.1, NetworkX 2.4 and scikit-learn 0.23.2, including successful `NPZData` import.

### Stronger validation-only saved-checkpoint check

If the selected LastFM dataset, split and checkpoint exist locally at their documented paths:

```bash
cd ~/Documents/H2GCN

docker run --rm --platform linux/amd64 \
  -v "$PWD":/workspace -w /workspace h2gcn-repro:clean-test \
  python scripts/verify_lastfm_checkpoint_validation_only.py
```

**Verified output (4 November 2026):** restored checkpoint weights matched the saved checkpoint and were repeatable; validation accuracy **85.4057%**, macro-F1 **63.6393%**, loss **1.08940**. The script verifies hashes of the graph, split, selection and checkpoint against the previously recorded final-test provenance. It does not train or compute held-out test accuracy. The test-evaluation script `evaluate_lastfm_h2gcn_selected_once.py --check-only` is deliberately unsuitable for this purpose after the one-time output already exists, because it refuses to overwrite or reopen the final-test folder.

Expected nonfatal warnings include NumPy divide-by-zero for zero-feature nodes during feature normalisation and CUDA initialization messages on CPU-only Docker setups. Preserve them in experiment logs where relevant.

**Scope:** this demonstrates clean-environment checkpoint compatibility for one documented LastFM validation task. It does not establish byte-identical dependencies, full training determinism, full-grid rerun fidelity or availability of excluded data to a new user.

### If the build fails

- **Image tag unavailable:** Inspect the public Python Docker image tags and choose a supported Python 3.8 / Debian variant. Record the exact replacement base and digest; do not claim the same image recipe was tested.
- **Old wheels or dependency-resolution error:** Compare the errors with `docker/provenance/working_pip_freeze.txt`. The older TensorFlow dependency graph may require explicit transitive versions. Record any adjustments in the Dockerfile and requirements file.
- **`exec format error`:** Ensure you used `--platform linux/amd64` for H2GCN and that Docker Desktop enables x86-64 emulation.
- **Out of memory or slow execution:** Increase the Docker memory allocation, reduce competing processes and use a single smoke-test graph. Large grids are not necessary to verify the environment.
- **Graph file not found:** The repo intentionally excludes the large NPZ files. Download/generate the data first, following `REPRODUCIBILITY.md`.
- **Unexpected metric differences:** Compare the exact split file, dataset checksum, model architecture, train seed, early-stopping policy and selected checkpoint. Do not tune on the test set to force agreement.

## 4. Build the separate GraphSAGE environment

`Dockerfile.graphsage` is already tracked in the repository and reflects the adapted GraphSAGE baseline used in the experiments.

On Apple Silicon:

```bash
cd ~/Documents/H2GCN

docker build --platform linux/arm64 \
  -f Dockerfile.graphsage -t graphsage-repro:pt251 .

docker run --rm --platform linux/arm64 graphsage-repro:pt251 \
  python -c 'import torch,numpy,scipy,sklearn; print(torch.__version__,numpy.__version__,scipy.__version__,sklearn.__version__)'
```

On an x86-64 Linux/Windows machine, do not silently claim the `arm64` image has been tested there. The underlying framework can be built for a different architecture, but that requires a separately validated build. The GraphSAGE adapter also includes a specific handling rule for isolated nodes, which is documented in `README_graphsage_setup.md`.

## 5. Access the source project and datasets

The repository contains the [upstream H2GCN source](https://github.com/GemsLab/H2GCN) and student-developed adapters. The public research release includes summary outputs and scripts, not the entire experimental data volume. Relevant sources:

- Released `syn-cora` source: upstream `npz-datasets/` and its README. Those NPZ archives do **not** preserve the original paper's split protocol; reconstructed splits must be disclosed.
- LastFM-Asia archive: [Stanford SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html), then `scripts/preprocess_lastfm.py` to create the project-specific processed NPZ and split.
- Constructed 30-graph benchmark: `scripts/generate_sbm_grid.py`, `scripts/generate_dcsbm_grid.py`, `scripts/calibrate_dcsbm.py`, and `scripts/audit_structural_grid.py`.

Read the script CLI directly (`python <script> --help` when available). Some workflows use additional locally generated files or paths, and **have not been verified end-to-end from a fresh checkout**.

## 6. Verified scope and remaining reproducibility limits

**Verified locally on 4 November 2026:**

1. Clean `linux/amd64` Docker build from `Dockerfile.h2gcn` completed without errors.
2. Installed package versions and original H2GCN NPZ dataset loader passed the independent check.
3. The frozen LastFM selected checkpoint restored repeatably, matched recorded dataset/split/checkpoint hashes and reproduced its validation metrics exactly at the reported precision.

**Not established by these checks:**

1. A bit-for-bit clone of the previous `h2gcn-repro:v1` environment. SciPy differs (1.4.1 versus 1.5.0).
2. Training from scratch on a new environment reproducing all reported synthetic and LastFM scores.
3. End-to-end data acquisition and experiment rerun directly from a bare GitHub clone; the larger graph NPZ archives and checkpoints remain excluded and require the documented preparation steps.

All Dockerfiles, documentation and verification scripts should be committed alongside the curated results. For any future experiment reproduction, compare the fixed graph and split, seeds, hyperparameter-selection criteria, checkpoint provenance and numeric precision before reporting an agreement or discrepancy.
