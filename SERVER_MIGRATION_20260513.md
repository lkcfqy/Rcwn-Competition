# RCWN Server Migration Handoff - 2026-05-13

This file is the quick recovery checklist for moving `/home/lkc/lkcproject/rcwn` to a new server.

## Current Best

- Legal online best: `65.9109`
- Latest submitted package: `submission/fqy_hat_hat_ssttta_275225500_raw.zip`
- Submission recipe: current native-io HAT-L 8-TTA `0.275` + previous HAT-L 8-TTA `0.225` + public `SSTXLarge_Plus_DFLIP_X2` 8-TTA `0.500`, raw.
- Local clean full120 metrics: `PSNR 33.98939 / SSIM 0.99801 / Edge 0.97756 / LPIPS 0.04577 / proxy 41.84101`
- Online gain: `+0.2222` over `65.6887`; distance to `66`: `0.0891`.

## GitHub Scope

GitHub should contain:

- source code under `src/`
- compact experiment CSV/TXT records under `experiments/`
- manifests under `manifests/`
- root markdown handoff/log files
- `submission/README.md`
- the latest submitted zip `submission/fqy_hat_hat_ssttta_275225500_raw.zip`

GitHub should not contain ordinary local-heavy artifacts:

- `超分竞赛数据集/`
- `external_models/`
- `checkpoints*/`
- `logs/`
- generated submission image folders
- `experiments/pred_cache_*`
- experiment output PNG/PT caches

Reason: this workspace is about `69G`; `external_models/` alone is about `58G`, and many checkpoint files exceed GitHub's normal `100MB` single-file limit. Use a separate disk copy, cloud bucket, or redownload public weights on the new server.

## Local Cleanup Performed

The local workspace was slimmed on 2026-05-13 from about `69G` to about `2.1G`.

Kept local-only assets:

- `超分竞赛数据集/` (`361M`)
- `external_models/SST/`
- `external_models/HF_weights/dslisleedh__SSTXLarge_Plus_DFLIP_X2/SSTXLarge_Plus_DFLIP_X2.pth`
- `checkpoints_hat_l_nativeio_official_cont_limit1680_lr8e8_full120clean_avg/`
- `checkpoints_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg/`
- `checkpoints_hat_l_official_cont_lr3e6_from_ep4_p48_avg/`
- `submission/fqy_hat_hat_ssttta_275225500_raw.zip`

Deleted local-only assets:

- closed external-model scans under `external_models/`, including the large `NTIRE2026_infraredSR`, `Real-IISR`, `EAMamba`, `ATD`, `SwinFIR`, `OpenModelDB_weights`, and related branches
- closed/negative checkpoint directories
- `experiments/*/` per-image metric dirs, prediction caches, and generated output images
- `logs/`
- old generated submission folders and old zip files

All deleted items are either recorded in CSV/markdown, rebuildable, redownloadable public weights, or closed branches. The current `65.9109` reproduction path is preserved.

## Minimal New-Server Restore

```bash
git clone https://github.com/lkcfqy/Rcwn-Competition.git rcwn
cd rcwn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then restore or redownload these local-only assets into the same paths:

- official dataset: `超分竞赛数据集/`
- current HAT checkpoint: `checkpoints_hat_l_nativeio_official_cont_limit1680_lr8e8_full120clean_avg/best_x2.pth`
- previous HAT checkpoint: `checkpoints_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`
- public SST repo/code: `external_models/SST/`
- public SST weight: `external_models/HF_weights/dslisleedh__SSTXLarge_Plus_DFLIP_X2/SSTXLarge_Plus_DFLIP_X2.pth`
- optional prior Mamba weight: `external_models/HF_weights/dslisleedh__MambaIRV2L_DFLIP_X2/MambaIRV2L_DFLIP_X2.pth`

## First Files To Read

1. `HANDOFF.md`
2. `PROJECT_LOG.md`
3. `experiments/submission_log.csv`
4. `experiments/probe_summary_20260510.csv`
5. `RESTART_PROMPT.md`
6. `rcwn.md`

## Latest Platform Fit

After the `65.9109` feedback:

- log: `logs/rank_platform_candidates_after_sst_submit_20260513.log` locally
- proxy fit: `score=-11.153564+1.842257*proxy`, RMSE `0.04407`
- ridge RMSE: `0.02041`
- top known family remains HAT/HAT/SST-TTA raw at about `ridge 65.8756 / proxy-fit 65.9283`

The log itself is local-only because `logs/` is ignored; the important numbers are recorded in `PROJECT_LOG.md` and `HANDOFF.md`.

## Continue From Here

- Do not spend submissions on small postprocess variants around the same SST-TTA family unless a new fit or online result changes the picture.
- Need roughly `+0.0891` online score to reach `66`.
- Practical next search should focus on a new legal public x2 model/weight that complements HAT/SST and improves the ridge view, especially without hurting LPIPS.
