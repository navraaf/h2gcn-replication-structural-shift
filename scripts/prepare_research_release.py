#!/usr/bin/env python3
"""Audit a curated, reproducible H2GCN project release.

Default: READ ONLY. --stage: stage only exact whitelisted files; never commit/push.
This script intentionally does not stage raw graph archives, model checkpoints,
training JSON dumps, Python backups, or directories recursively.
"""
from __future__ import print_function
import argparse
from pathlib import Path
import subprocess
import sys

CORE_SCRIPTS = [
    'scripts/audit_structural_grid.py',
    'scripts/gcn_npz_tf2.py',
    'scripts/graphsage_npz_reference.py',
    'scripts/run_gcn_validation_sweep.sh',
    'scripts/run_graphsage_full_grid.sh',
    'scripts/summarize_gcn_validation_sweep.py',
    'scripts/summarize_graphsage_grid.py',
    'scripts/summarize_three_model_degree_robustness.py',
    'scripts/gcn_npz_with_predictions.py',
    'scripts/graphsage_npz_with_predictions.py',
    'scripts/run_selected_degree_predictions.py',
    'scripts/extract_h2gcn_node_predictions.py',
    'scripts/run_h2gcn_prediction_extraction.sh',
    'scripts/run_h2gcn_remaining_grid.sh',
    'scripts/train_lastfm_h2gcn_validated.py',
    'scripts/lastfm_gcn_mlp_validated.py',
    'scripts/run_lastfm_h2gcn.sh',
    'scripts/run_lastfm_followup.sh',
    'scripts/run_lastfm_three_model_repeats.sh',
    'scripts/train_lastfm_h2gcn_sensitivity.py',
    'scripts/run_lastfm_h2gcn_sensitivity.sh',
    'scripts/summarize_lastfm_h2gcn_sensitivity.py',
    'scripts/evaluate_lastfm_h2gcn_selected_once.py',
    'scripts/train_lastfm_h2gcn_tuned_repeat.py',
    'scripts/run_lastfm_h2gcn_tuned_repeats.sh',
    'scripts/summarize_lastfm_h2gcn_tuned_repeats.py',
    'scripts/audit_lastfm_class_performance.py',
    'scripts/summarize_lastfm_models.py',
    'scripts/prepare_research_release.py',
    'Dockerfile.graphsage',
    'REPRODUCIBILITY.md',
    'RELEASE_CHECKLIST.md',
]
CORE_RESULTS = [
    'results/structural_grid_runs.csv',
    'results/structural_grid_summary.csv',
    'results/h2gcn_grid_runs.csv',
    'results/h2gcn_grid_summary.csv',
    'results/gcn_all_validation_candidates.csv',
    'results/gcn_selected_by_validation.csv',
    'results/graphsage_all_validation_candidates.csv',
    'results/graphsage_selected_by_validation.csv',
    'results/h2gcn_gcn_graphsage_paired_runs.csv',
    'results/degree_robustness_accuracy_checks.csv',
    'results/degree_robustness_three_model_runs.csv',
    'results/degree_robustness_three_model_summary.csv',
    'results/degree_robustness_three_model_composition.csv',
    'results/lastfm_comparison/lastfm_model_runs.csv',
    'results/lastfm_comparison/lastfm_model_summary.csv',
    'results/lastfm_h2gcn_sensitivity/validation_sensitivity_summary.csv',
    'results/lastfm_h2gcn_sensitivity/validation_selected_config.json',
    'results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/selected_test_results.json',
    'results/lastfm_h2gcn_sensitivity/selected_l2small_final_test/test_class_metrics.csv',
    'results/lastfm_h2gcn_tuned_repeats/tuned_lastfm_h2gcn_runs.csv',
    'results/lastfm_h2gcn_tuned_repeats/tuned_lastfm_h2gcn_summary.csv',
]
OPTIONAL_DOCS = [
    'README_full_grid.md',
    'README_graphsage_setup.md',
    'README_selected_degree_predictions.md',
    'README_LastFM_H2GCN.md',
    'README_LastFM_followup.md',
    'README_LastFM_H2GCN_sensitivity.md',
    'README_LastFM_selected_final_test.md',
    'README_lastfm_tuned_repeats.md',
    'README_H2GCN_prediction_extraction.md',
]
MAX_EACH_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 18 * 1024 * 1024


def git(*args):
    return subprocess.run(['git'] + list(args), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', action='store_true', help='Explicitly stage only verified curated files')
    ap.add_argument('--allow-missing-core', action='store_true',
                    help='Do not stage missing paths; still report them (not recommended)')
    args = ap.parse_args()
    root = git('rev-parse', '--show-toplevel')
    if root.returncode != 0:
        sys.exit('ERROR: Not inside a Git repository.')
    repo = Path(root.stdout.strip()).resolve()
    print('Repository:', repo)
    remote = git('remote', 'get-url', 'origin')
    print('Origin:', remote.stdout.strip() if remote.returncode == 0 else '(not configured)')
    tracked = git('ls-files', '-z')
    tracked_set = set(tracked.stdout.split('\0')) if tracked.returncode == 0 else set()
    critical = CORE_SCRIPTS + CORE_RESULTS
    wanted = critical + OPTIONAL_DOCS
    missing_core = []
    ignored = []
    oversize = []
    available = []
    optional_missing = []
    for rel in wanted:
        path = repo / rel
        if not path.is_file():
            if rel in critical:
                missing_core.append(rel)
            else:
                optional_missing.append(rel)
            continue
        try:
            path.resolve().relative_to(repo)
            inside_repo = True
        except ValueError:
            inside_repo = False
        if path.is_symlink() or not inside_repo:
            ignored.append(rel + ' (symlink or outside repository)')
            continue
        length = path.stat().st_size
        if length > MAX_EACH_BYTES:
            oversize.append((rel, length))
            continue
        if rel not in tracked_set:
            check = git('check-ignore', '-q', '--', rel)
            if check.returncode == 0:
                ignored.append(rel + ' (.gitignore)')
                continue
        available.append((rel, length))
    total = sum(size for _, size in available)
    print('\nCore required:', len(critical))
    print('Files eligible to stage (including available optional docs):', len(available))
    print('Combined size: {:.2f} MiB'.format(total / 1048576.0))
    print('Already tracked among eligible:', sum(1 for p, _ in available if p in tracked_set))
    print('\nMISSING CORE FILES (must inspect):')
    for item in missing_core:
        print('  MISSING', item)
    if not missing_core:
        print('  None')
    print('\nIGNORED/UNSAFE FILES:')
    for item in ignored:
        print('  BLOCKED', item)
    if not ignored:
        print('  None')
    print('\nOVERSIZE FILES:')
    for item, size in oversize:
        print('  BLOCKED', item, size, 'bytes')
    if not oversize:
        print('  None')
    if optional_missing:
        print('\nOptional docs missing:', ', '.join(optional_missing))
    print('\nExact eligible files:')
    for item, size in available:
        print('  {:>8}  {}'.format(size, item))
    if total > MAX_TOTAL_BYTES:
        print('\nBLOCKED: total curated release size exceeds 18 MiB.')
        return 2
    if not args.stage:
        print('\nDRY RUN ONLY. No files changed or staged.')
        print('Review the report, then use --stage if satisfied.')
        return 0
    if (missing_core and not args.allow_missing_core) or ignored or oversize:
        print('\nREFUSE STAGE: critical missing, ignored or oversized files require review.')
        return 2
    before = git('diff', '--cached', '--name-only')
    if before.returncode == 0 and before.stdout.strip():
        print('\nWARNING: some unrelated files are already staged; review them too:')
        print(before.stdout.strip())
    rels = [rel for rel, _ in available]
    # Only exact listed paths. Never git add -A, '.', or an entire results directory.
    out = git('add', '--', *rels)
    if out.returncode != 0:
        print('ERROR: git add failed: ', out.stderr)
        return 3
    print('\nSTAGED curated files. No commit or push performed.')
    summary = git('diff', '--cached', '--stat')
    if summary.returncode == 0:
        print(summary.stdout)
    check = git('diff', '--cached', '--check')
    if check.returncode != 0:
        print('WARNING: git diff --cached --check found issues:\n' + check.stdout + check.stderr)
        return 4
    print('PASS: git diff --cached --check')
    print('Review: git diff --cached --name-only')
    return 0


if __name__ == '__main__':
    sys.exit(main())
