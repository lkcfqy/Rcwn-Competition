# RCWN Stage-1 Super-Resolution Workspace

This repo is the working directory for the RCWN competition, focused on stage-1 `x2` legal score pushing under the rules in [rcwn.md](/home/lkc/lkcproject/rcwn/rcwn.md:1).

## Current Status

- Best confirmed legal online score: `65.4881`
- Best confirmed local clean `full120` proxy: `41.63416`
- Fixed `probe40` baseline: `43.08219`
- Current promising line: `native-io HAT-L`, which has now become the current clean-best family; `official-only p96` is the previous incumbent baseline
- Latest completed HAT log: `logs/train_x2_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg.log`
- Latest legal HAT-HAT ensemble sweep: best `alpha_a=0.9` reached only `proxy 41.63031`, which is now below the current clean best and not a submission candidate
- Latest `native-io HAT-L` probe: `logs/train_x2_hat_l_nativeio_official_probe40_limit512_p96_lr2e7_e1_avg.log`, final `probe40 43.09142`, which is `+0.00923` over the fixed `43.08219` baseline
- Latest `native-io HAT-L` clean run: `logs/train_x2_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg.log`, final `full120 proxy 41.63416`, which is `+0.00391` over the previous incumbent `41.63025`
- Current runtime status: no active training or evaluation process; the next legal continuation should start from `checkpoints_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`
- New self-designed line status: both `ThermalEdgeSR tiny` from-scratch and `ThermalEdgeSR small + HAT distillation` official `probe40` runs closed negative at `logs/train_x2_thermal_edge_tiny_probe40_limit512_p96.log` and `logs/train_x2_thermal_edge_small_kdhat_probe40_limit512_p96.log`

Current strategy:

- keep one low-risk legal slow-crawl line alive to steadily raise the current best
- but treat large-jump model or training-signal discovery as the main objective for any realistic `70+` attempt
- do not spend submissions on ordinary `+0.00x` local gains; only package when there is clearly stronger evidence than the current `65.5x` band

The current best confirmed clean `full120` checkpoint is:

- `checkpoints_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`

The previous clean incumbent checkpoint is:

- `checkpoints_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`

The older clean baseline checkpoint is:

- `checkpoints_hat_l_official_cont_lr3e6_from_ep4_p48_avg/best_x2.pth`

Its clean `full120` metrics are:

- `PSNR 33.78392 / SSIM 0.99789 / Edge 0.97648 / LPIPS 0.04503 / proxy 41.63416`

Important note:

- `val_probe40_seed42` is a subset of `val_full120_seed42`, so `probe40`-trained `p96` runs overlap `80/120` images with `full120` and their `41.62784/41.62860` numbers are exploratory, not clean confirmation

## Repo Layout

- `src/`: training, evaluation, inference, dataset prep, route analysis, and model adapters
- `manifests/`: frozen eval splits such as `smoke4`, `probe40`, and `full120`
- `experiments/`: compact CSV records, probe summaries, source-match tables, and selected text manifests
- `HANDOFF.md`: current battle state for resuming quickly
- `PROJECT_LOG.md`: chronological decisions and experiment record
- `RESTART_PROMPT.md`: ready-to-paste prompt for a fresh conversation
- `submission/README.md`: writeup used inside legal submission packages

## Tracked Vs Local-Only

Track in Git:

- source code under `src/`
- frozen manifests under `manifests/`
- compact experiment records under `experiments/`
- root markdown docs and `requirements.txt`

Keep local-only:

- `超分竞赛数据集/`
- `external_data/`
- `external_models/`
- `checkpoints*/`
- `logs/`
- `submission/*.zip`
- virtual environments such as `.venv/` and `.venv_h5/`

See [GITHUB_SYNC.md](/home/lkc/lkcproject/rcwn/GITHUB_SYNC.md:1) for cross-machine guidance.

## Environment

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Some model families also expect local public repos or weights under `external_models/`, which is intentionally ignored by Git.

## Resume Work

For a fresh chat or a different machine:

1. Read `HANDOFF.md`
2. Read `PROJECT_LOG.md`
3. Read the latest `experiments/probe_summary_*.csv`
4. Check whether a training or eval process is still running
5. Continue from the currently active legal main line, not from already closed branches

The ready-to-paste restart text is in [RESTART_PROMPT.md](/home/lkc/lkcproject/rcwn/RESTART_PROMPT.md:1).
