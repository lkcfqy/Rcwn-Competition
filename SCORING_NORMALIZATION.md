# RCWN scoring normalization notes

This note transcribes the normalization rule obtained from the competition scoring description and records how it affects the current x2 stage-1 strategy.

## Normalization Formula

For "higher is better" metrics such as PSNR, SSIM, and edge preservation:

```text
s = (x - x_min) / (x_max - x_min)
```

For "lower is better" metrics such as LPIPS, parameter count, and model size:

```text
s = (x_max - x) / (x_max - x_min)
```

If the normalized value is outside the valid interval, it is clipped:

```text
s = min(1, max(0, s))
```

The per-item score is:

```text
score_i = s_i * w_i
```

The final automatic score is the sum of all per-item scores.

## Stage-1 Weights

For the x2 first stage, `rcwn.md` says only image quality metrics are scored:

| Metric | Direction | Weight |
| --- | --- | ---: |
| PSNR | higher is better | 30 |
| SSIM | higher is better | 30 |
| LPIPS | lower is better | 20 |
| Edge preservation | higher is better | 20 |

Parameter count, model size, and inference time are not score items in stage 1. They matter for later stages and constraints, but not for the current x2 leaderboard score.

## Practical Implications

- The current local proxy `PSNR + 6 * SSIM + 2 * Edge - 2 * LPIPS` is only a correlation proxy, not the platform formula.
- To compute a real platform-like local score, we still need the hidden or official `x_min` and `x_max` for PSNR, SSIM, LPIPS, and edge preservation.
- Once `x_min/x_max` are known, marginal score gain is:

```text
higher-better metric: delta_score = w * delta_x / (x_max - x_min)
lower-better metric:  delta_score = w * (-delta_x) / (x_max - x_min)
```

- This can change candidate ranking. For example, a small LPIPS drop can be worth more than a small PSNR rise if the LPIPS normalization range is narrow.
- Clipping means improvements beyond `x_max` or below `x_min` stop contributing to the official score.

## Current x2 Strategy Impact

This formula does not by itself create a new `66/67` path, but it is useful for triage:

- Prioritize candidates that improve all four quality metrics, not only the current proxy.
- Do not favor smaller models for stage 1 unless their image metrics are competitive, because model size and parameter count are not stage-1 score items.
- Keep the fixed low-Mamba HAT/HAT/MambaIRv2 ensemble as the current legal best because it produces the strongest confirmed online score so far: `65.6887`.
- Revisit local candidate ordering if official or reliably inferred `x_min/x_max` values become available.

## Current Empirical Fit

Until the real `x_min/x_max` values are known, the best available calibration is still the legal submission history.

Using the 10 legal submitted packages with platform scores and excluding archived nearest/retrieval rows, a one-variable fit gives:

```text
platform_score ~= -11.246223 + 1.844556 * local_proxy
RMSE ~= 0.0458
```

This fit is not the official formula, and it is too coarse for deciding between `+0.001` candidates. It is still useful for scale: moving from the current submitted fixed low-Mamba `41.72854` proxy to `66.0` online likely needs roughly `+0.149` local proxy, far larger than the current HAT polishing gains.

Concrete thresholds from this fit:

```text
66.0 online ~= local_proxy 41.87795
67.0 online ~= local_proxy 42.42009
```

The current best offline legal full120 candidate is `41.73174` from the three-way current-HAT / previous-HAT / MambaIRv2-Large raw ensemble. The gap to the estimated `66` line is therefore still about `+0.146`. This is why HAT-HAT, MambaIRv2 ensemble, and oracle/router polishing should be treated as possible tiny leaderboard nudges, not real `66/67` paths.

## Candidate Reranking Tool

`src/rank_platform_candidates.py` now scans existing candidate CSVs and ranks them with both the one-variable proxy fit and a four-metric ridge fit from submitted legal packages:

```bash
python3 src/rank_platform_candidates.py --candidate experiments --require-text full120 --top-k 12
```

The current ridge fit is intentionally only a triage tool:

```text
lambda = 0.1
RMSE ~= 0.01954
standardized coefficients:
  intercept 64.490360
  PSNR      +0.403518
  SSIM      +0.415485
  Edge      +0.377963
  LPIPS     -0.224706
```

This reranking changes the best marginal HAT-HAT polishing candidate:

- Proxy best: `alpha=0.65 + cubic blend=0.015`, `proxy 41.72349`
- Ridge platform-style best before the low-Mamba refresh: `alpha=0.65 + cubic blend=0.005`, `proxy 41.72199`, `LPIPS 0.04566`, updated ridge estimate `65.6960`

The `blend=0.005` route is a plausible low-risk small-refresh candidate if another submission slot becomes available and no stronger model appears. It still does not close the estimated `66` gap.

After the MambaIRv2 and three-way ensemble refresh, the proxy-best local candidate is:

- current native-io HAT 8-TTA `0.60` + previous non-native HAT 8-TTA `0.30` + MambaIRv2-Large non-TTA `0.10`, raw full120 `41.73174`

The one-variable proxy fit estimates this around `65.7303`, still below `66`; the updated ridge fit ranks it lower than the submitted fixed low-Mamba route because its LPIPS is worse.

After the low-Mamba sweep, the current ridge-first fixed candidate is:

- current native-io HAT 8-TTA `0.60` + previous non-native HAT 8-TTA `0.35` + MambaIRv2-Large non-TTA `0.05`, raw full120 `41.72854`, LPIPS `0.04601`, updated ridge estimate `65.7003`

It was submitted on 2026-05-12 and returned `65.6887`, so it is the current confirmed legal online best. After incorporating that feedback, the k4 ridge router is close but lower at ridge `65.6994`, while the HAT-HAT `alpha=0.65 + cubic blend=0.005` route is around `65.6960`. These are still small-refresh options rather than a decisive `66` path.
