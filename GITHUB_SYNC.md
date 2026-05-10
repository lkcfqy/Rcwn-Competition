# GitHub Sync Guide

This repo is usable across computers, but only if we keep GitHub focused on source code, manifests, and compact experiment records.

## Track In Git

- `src/`
- `manifests/`
- `experiments/*.csv`
- `HANDOFF.md`
- `PROJECT_LOG.md`
- `README.md`
- `RESTART_PROMPT.md`
- `rcwn.md`
- `requirements.txt`
- `.gitignore`

Recommended experiment records to keep:

- `experiments/submission_log.csv`
- `experiments/probe_summary_*.csv`
- `experiments/source_match_*.csv`
- any small summary CSV that captures a decision or leaderboard-relevant result

## Do Not Track In Git

- `超分竞赛数据集/`
- `external_data/`
- `external_models/`
- `checkpoints*/`
- `logs/`
- `submission/*.zip`
- generated submission image folders
- local virtual environments such as `.venv/` and `.venv_h5/`

These are either too large, easy to regenerate, license-sensitive, or machine-local.

## Current Important Local-Only Assets

If you move to a new machine and want to continue immediately, these should be copied separately or redownloaded:

- official dataset under `超分竞赛数据集/`
- public pretrained repos under `external_models/`
- active checkpoints under:
  - `checkpoints_hat_l_official_cont_lr3e6_from_ep4_p48_avg/`
  - `checkpoints_hat_l_official_p96_cont_limit1024_lr2e7_avg/`
  - `checkpoints_hat_l_official_p96_cont_limit1800_lr15e7_avg/`
- any local submission outputs you still care about under `submission/`

## Practical Workflow

1. Commit code, manifests, and compact CSV/markdown records to GitHub.
2. Keep large assets ignored.
3. On a new machine:
   - clone the repo
   - recreate the Python environment
   - install `requirements.txt`
   - restore or redownload datasets and public model repos into the same ignored paths
   - copy back the active checkpoints if you want to resume without retraining
4. Start by reading:
   - `HANDOFF.md`
   - `PROJECT_LOG.md`
   - `experiments/submission_log.csv`
   - latest `experiments/probe_summary_*.csv`
   - `RESTART_PROMPT.md`

## Important Caveat

`.gitignore` only affects untracked files. If a large file or dataset was already added to Git before, you must remove it from the Git index before GitHub upload, otherwise ignore rules will not help.
