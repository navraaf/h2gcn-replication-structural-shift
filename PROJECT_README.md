# H2GCN Replication and Structural Distribution Shift

COMP8240 Applications of Data Science
Macquarie University, 2026

## Research objective
Reproduce H2GCN and investigate whether its performance
under heterophily remains robust when graph structure changes
at approximately comparable homophily levels.

## Original research
Zhu et al. (2020), Beyond Homophily in Graph Neural Networks:
Current Limitations and Effective Designs.

Original implementation:
https://github.com/GemsLab/H2GCN

## Our contributions
- Reconstructed NPZ data loading and experimental splits.
- Reproduced H2GCN-2 on syn-cora at h approximately 0.10.
- Preprocessed LastFM-Asia.
- Generated 15 SBM and 15 DC-SBM synthetic graphs.
- Analysed structural differences between graph families.

## Experimental status
Original-paper replication: completed approximately.
New graph construction: completed.
LastFM compatibility testing: completed.
Full benchmark experiments on new datasets: pending.

## Project files
scripts/ - Dataset construction and analysis scripts.
results/ - Summary CSVs.
constructed_data/ - Graph-generation summaries.
h2gcn/datasets/npz.py - Custom NPZ loader.

Downloaded datasets and local model checkpoints are excluded
from the public repository.

The final reproducibility instructions and benchmark results
will be added after experimental verification.
