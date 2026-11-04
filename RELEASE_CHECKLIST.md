# Research repository release checklist

**Based on the local audit of 4 November 2026.** The audit showed many untracked scripts and `results/` summaries. It did not establish that the remote GitHub branch contains them.

## Critical before the final submission

- [ ] Add the currently untracked experiment scripts, controlled generator code and curated CSV/JSON summaries to Git. Use `python3 scripts/prepare_research_release.py` **first**, then explicitly `--stage` after review. The tool never commits or pushes.
- [ ] Verify `git diff --cached --stat`, `git diff --cached --name-only`, and `git diff --cached --check` before committing.
- [ ] Ensure that `.gitignore` excludes checkpoints (`*.index`, `*.data-*`), Docker logs, temporary files, original raw datasets, large processed NPZ graphs, and `*.backup` scripts. **Do not automatically add all of `results/`.**
- [ ] Verify that a fresh clone can obtain or rebuild the base `h2gcn-repro:v1` Docker image. The current audit showed `Dockerfile.graphsage` but did not establish the base image build recipe. Record exact package versions and build commands.
- [ ] Add reproducible instructions for obtaining Cora-derived feature/label source NPZ and LastFM-Asia raw files, along with exact generator CLI, random seeds, fixed split and output filenames. Verify that the constructed graphs can be regenerated from these inputs. Do not claim clean-room reproducibility before testing it.
- [ ] Confirm whether `PROJECT_README.md` is visible and linked from the GitHub default `README.md`, which currently appears to be the original upstream README. Update the project landing page with a clear link to `REPRODUCIBILITY.md`, research title, and report attachment. Do not overwrite upstream attribution.
- [ ] Verify summary CSV totals (30 graphs, 10 conditions, 30 selected H2GCN, 30 GCN and 30 GraphSAGE runs; LastFM three training seeds) and inspect the committed outputs for accidental sensitive data.
- [ ] Commit and push the reviewed changes. After pushing, check the web repository and copy its public HTTPS URL into the report's required footnote. **The local audit did not verify public GitHub visibility.**

## Evidence to report accurately

- [ ] H2GCN's published syn-cora h≈0.1 result is only approximately replicated because the original experiment split identifiers are unavailable.
- [ ] SBM versus DC-SBM differs in degree distribution, isolation rate and connectivity: avoid causal attribution to degree CV alone.
- [ ] The node-level decomposition is descriptive. Three graph realisations, rather than individual nodes, are the experimental replication units for synthetic graphs.
- [ ] GCN and GraphSAGE use validation-selected settings on each synthetic graph; H2GCN uses a fixed setting there.
- [ ] LastFM compares three *initialisation seeds* on one fixed graph/split. H2GCN was tuned by validation; GCN and MLP have different tuning budgets.
- [ ] LastFM tuned H2GCN has higher accuracy but not equal recall for all 18 classes; include macro-F1 and per-class limitations.
- [ ] Keep the core report within **10 pages**, excluding references, and include original paper justification, original dataset, replication, constructed dataset and method, quantitative results, ethics, code/reproducibility, and limitations.

## Local commands after review (do not run blindly)

```bash
cd ~/Documents/H2GCN
python3 scripts/prepare_research_release.py
# Review missing files or blockers, then:
python3 scripts/prepare_research_release.py --stage
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check
# Only after reviewing the staged changes:
git commit -m "Add structural-shift experiments and reproducibility evidence"
git push origin HEAD
```

**Caution:** `git push` transfers the staged commit to the remote. Check the full staged file listing first. If the working tree contains unrelated changes, the script stages only its exact allowlist; it does not wipe existing staging.
