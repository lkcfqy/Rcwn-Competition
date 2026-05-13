# RCWN Stage-1 Super-Resolution Workspace

This repo is the working directory for the RCWN competition, focused on stage-1 `x2` legal score pushing under the rules in [rcwn.md](/home/lkc/lkcproject/rcwn/rcwn.md:1).

## Current Status

- Best confirmed legal online score: `65.9109`
- Best confirmed local clean raw `full120` proxy: `41.63526`
- Best confirmed local clean `full120 + blend_interp=0.02` proxy: `41.64162`
- Latest successful submission: `submission/fqy_hat_hat_ssttta_275225500_raw.zip`, current native-io HAT 8-TTA `0.275` + previous HAT 8-TTA `0.225` + SSTXLarge_Plus_DFLIP_X2 `0.500`, raw full120 `41.84101`, online `65.9109`
- Best offline clean TTA/ensemble proxy: `41.73174` from current native-io HAT 8-TTA `0.60` + previous non-native HAT 8-TTA `0.30` + official MambaIRv2-Large x2 non-TTA `0.10`, raw full120; not packaged because it is only `+0.00320` over the submitted fixed low-Mamba local proxy and the updated platform ridge fit ranks it lower
- Latest submitted platform-calibrated fixed candidate: current native-io HAT 8-TTA `0.275` + previous non-native HAT 8-TTA `0.225` + SSTXLarge_Plus_DFLIP_X2 `0.500`, raw full120 `proxy 41.84101`, `LPIPS 0.04577`, online `65.9109`
- Best platform-calibrated routed candidate: k4 LR-cluster ridge router, 75 images HAT-HAT `blend=0.005` and 25 images three-way `0.60/0.30/0.10`, full120 `proxy 41.72627`, `LPIPS 0.04589`, updated ridge estimate `65.6994`
- Previous HAT+Mamba offline proxy best: `41.72977` from current native-io HAT 8-TTA `alpha=0.90` + official MambaIRv2-Large x2 non-TTA `alpha=0.10`, raw full120
- Previous HAT-HAT offline proxy best: `41.72349` from current native-io HAT `alpha=0.65` + previous non-native HAT `alpha=0.35`, 8-TTA, `cubic blend=0.015`
- Best platform-calibrated marginal HAT-HAT candidate: same HAT-HAT combo with `alpha=0.65`, `cubic blend=0.005`, full120 `proxy 41.72199`, `LPIPS 0.04566`, updated ridge estimate `65.6960`; this is below the submitted fixed low-Mamba candidate and not a `66`-level signal
- Fixed `probe40` baseline: `43.08219`
- Current promising line: `native-io HAT-L`, but ordinary continuation is frozen because the slope is only `+0.00x`
- Latest completed HAT log: `logs/train_x2_hat_l_nativeio_official_trainall1800_lr8e8_p96_avg.log`
- Latest legal HAT-HAT ensemble sweep: best `alpha_a=0.9` reached only `proxy 41.63031`, which is now below the current clean best and not a submission candidate
- Latest `native-io HAT-L` probe: `logs/train_x2_hat_l_nativeio_official_probe40_limit512_p96_lr2e7_e1_avg.log`, final `probe40 43.09142`, which is `+0.00923` over the fixed `43.08219` baseline
- Latest `native-io HAT-L` clean continuation: `checkpoints_hat_l_nativeio_official_cont_limit1680_lr8e8_full120clean_avg`, final raw `full120 proxy 41.63526`; best legal blend002 proxy `41.64162`
- Current runtime status: no active training or evaluation process; latest submitted package is the fixed low-Mamba small-refresh package
- Latest public-weight screening: ONNX/OpenModelDB/HF/SwinFIR/MAN Stage A produced no candidate above `45.20`; best public single in the latest batch is HATFIR `44.94415` followed by HF standard HAT x2 `44.66887`, and best HAT+external smoke remains below HAT-HAT (`HAT+HATFIR 45.49817` vs HAT-HAT `45.51113`)
- Latest LR-only sanity: `blend002 + cubic` smoke4 reached `45.39291`, and native-io HAT TTT topped out at `45.39207`; both are polishing only, not submission signals
- Latest HAT TTA subset sanity: `s247=0,1,2,4,5,6,7` with `cubic blend=0.01` reached smoke4 `45.49682`, but full120 fell to `41.72006` versus full8 `41.72060`; branch closed
- Latest highpass residual sanity: HATFIR `45.49527`, CAT-A `45.49751`, and HAT-HAT `45.50097`, all below their ordinary alpha blends and below HAT-HAT alpha `45.51113`; branch closed
- Latest extra legal probes: x4-internal-to-x2 downsample (`DAT_2_x4` best `44.63012`, SwinIR x4 best `44.42676`), HAT SWA (`45.49424` best), HAT refiner (`45.38152`), and tile inference (`45.33157`) all failed to beat the current HAT TTA/HAT-HAT baselines; no new package candidate
- Latest thermal-domain public check: `CallMeDaniel/NTIRE2026-InfraredSR` RFRSR x4 weight is downloaded but blocked by unavailable inference repo/custom PDA-DCN code; `Kronbii/thermal-super-resolution` has no released weight file in the repo
- Latest public single-model full120 check: HATFIR official x2 scored only `41.35036`; HAT TTA vs HATFIR full120 oracle is `41.72650` but LR-cluster router selects no HATFIR route, so it is not a package candidate
- Latest cold OpenModelDB checks: Swift-SRGAN x2 `42.20725`, realSR SwinIR GAN x2 `38.83051`; both are below Stage A and closed. `2x-PSNR` Google Drive link could not be fetched by `gdown`
- Latest metric-loss continuation sanity: `SSIM/Edge/LPIPS` full120 probe fell `41.63526 -> 41.63219`; milder `SSIM/Edge` probe40 fell `43.09584 -> 43.09211`; metric-loss branch is closed
- Scoring normalization note saved in [SCORING_NORMALIZATION.md](/home/lkc/lkcproject/rcwn/SCORING_NORMALIZATION.md:1): stage-1 score uses normalized PSNR/SSIM/LPIPS/Edge with weights `30/30/20/20`; parameter count and model size are not stage-1 score items
- New self-designed line status: both `ThermalEdgeSR tiny` from-scratch and `ThermalEdgeSR small + HAT distillation` official `probe40` runs closed negative at `logs/train_x2_thermal_edge_tiny_probe40_limit512_p96.log` and `logs/train_x2_thermal_edge_small_kdhat_probe40_limit512_p96.log`

Current strategy:

- current best already beats the known `65.6323`; latest legal online best is `65.9109`
- next legal target is `66`, with offline work focused on TTA/router/ensemble or a new x2 model signal before the next submission window
- latest three-way HAT/HAT/MambaIRv2 sweep refreshed local clean best to `41.73174`, but this is still a polishing-scale gain rather than a `66`-level signal
- empirical submission fit after the `65.6887` feedback estimates `66` needs local proxy around `41.87795`, about `+0.146` above the current offline best; `67` would need around `42.42009`
- `src/rank_platform_candidates.py` can rerank existing full120 candidates using submitted-platform feedback; it currently favors the fixed low-Mamba `0.60/0.35/0.05 raw` candidate by ridge, while the three-way `0.60/0.30/0.10 raw` remains proxy best
- full120 postprocess/alpha oracle is only `41.72830~41.73022`, so complex LR-only router has too little headroom for `66`
- freeze ordinary HAT continuation after the `65.5414` train-all result confirmed only a small online gain
- clean A/B returned `65.5408`, slightly below train-all `65.5414` and below the `65.56` gate
- spend remaining GPU/submissions on high-odds public x2 backbone/weight screening; only package new candidates at `full120 >= 41.66` or similarly strong `probe40`
- do not advance ONNX/OpenModelDB candidates unless Stage A is at least `45.20`

The current best confirmed clean `full120` checkpoint is:

- `checkpoints_hat_l_nativeio_official_cont_limit1680_lr8e8_full120clean_avg/best_x2.pth`

The previous clean incumbent checkpoint is:

- `checkpoints_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`

The older clean baseline checkpoint is:

- `checkpoints_hat_l_official_cont_lr3e6_from_ep4_p48_avg/best_x2.pth`

Its clean `full120` metrics are:

- raw: `PSNR 33.78492 / SSIM 0.99789 / Edge 0.97648 / LPIPS 0.04500 / proxy 41.63525`
- blend002: `PSNR 33.79283 / SSIM 0.99790 / Edge 0.97650 / LPIPS 0.04581 / proxy 41.64162`

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
- `SCORING_NORMALIZATION.md`: transcribed normalization formula and current scoring implications

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
