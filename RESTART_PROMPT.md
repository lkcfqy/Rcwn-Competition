# 新对话启动提示词

我在 `/home/lkc/lkcproject/rcwn` 做 RCWN 比赛第一阶段 `x2` 合法冲分，现实目标是在严格遵守 `rcwn.md` 规则的前提下继续冲线上 `66`，若出现强信号再争取 `67`。当前已超过已知 `65.6323`。请先读取：

- `HANDOFF.md`
- `PROJECT_LOG.md`
- `experiments/probe_summary_20260510.csv`
- `experiments/submission_log.csv`
- `submission/README.md`
- `rcwn.md`

然后先检查当前是否还有训练或评估进程在跑：

```bash
pgrep -af '[t]rain_hat_gray|[t]rain_generic_gray|[e]valuate_hat_gray|src/[t]rain|src/[e]valuate|src/[m]ake_submission' || true
```

如果发现活跃进程，再读取这些最新日志：

- `logs/train_x2_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg.log`
- `logs/train_x2_hat_l_official_p96_cont_limit1024_lr2e7_full120clean_avg.log`
- `logs/train_x2_hat_l_nativeio_official_probe40_limit512_p96_lr2e7_e1_avg.log`
- `logs/train_x2_hat_l_nativeio_official_cont_limit1024_lr2e7_full120clean_avg.log`
- `logs/train_x2_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg.log`

当前关键状态：

- 2026-05-13 最新线上 best 已更新：
  `submission/fqy_hat_hat_ssttta_275225500_raw.zip`
  current native-io HAT-L 8-TTA `0.275` + previous HAT-L 8-TTA `0.225` + public SSTXLarge_Plus_DFLIP_X2 8-TTA `0.500`，raw；
  clean full120 `PSNR 33.98939 / SSIM 0.99801 / Edge 0.97756 / LPIPS 0.04577 / proxy 41.84101`；
  已提交并返回线上 `65.9109`，相对上一合法 best `65.6887` 为 `+0.2222`，距离 `66` 还差 `0.0891`；
  该提交包已校验为单一顶层目录、`100` 张 `640x512` 8-bit grayscale PNG。
- 2026-05-13 用 `65.9109` 反馈重跑 `src/rank_platform_candidates.py`：
  `logs/rank_platform_candidates_after_sst_submit_20260513.log`；
  top 仍是 HAT/HAT/SST-TTA raw 同族，ridge 约 `65.8756`、proxy-fit 约 `65.9283`，当前尚无新的 `66` 强信号。
- 当前最好已确认合法线上分：`65.9109`
- 当前最好已确认本地 clean raw `full120`：`41.63526`
- 当前最好已确认本地 clean `full120 + blend_interp=0.02`：`41.64162`
- 当前最新成功提交：
  `submission/fqy_hat_hat_mambair_lowmamba_603505_raw.zip`
  fixed low-Mamba：current native-io HAT 8-TTA `0.60` + previous HAT 8-TTA `0.35` + MambaIRv2-Large `0.05`，raw clean full120 `41.72854`
  已提交并返回线上 `65.6887`，超过已知 `65.6323`，相对上一 best `65.6804` 为 `+0.0083`。
- 当前最新离线 best：
  当前 native-io clean HAT 8-TTA `0.60` + 上一版非 native-io clean HAT 8-TTA `0.30` + 官方 MambaIRv2-Large x2 非 TTA `0.10`，raw clean full120 `41.73174`；
  相对已提交 fixed low-Mamba 本地 `41.72854` 为 `+0.00320`，相对 HAT+Mamba `41.72977` 为 `+0.00197`，但 LPIPS 更差，更新后的 ridge 口径不如已提交包，仍不是 `66` 强信号。
- 当前平台拟合第一小刷候选：
  当前 native-io clean HAT 8-TTA `0.60` + 上一版非 native-io clean HAT 8-TTA `0.35` + 官方 MambaIRv2-Large x2 非 TTA `0.05`，raw clean full120 `41.72854`；
  `PSNR 33.87895 / SSIM 0.99797 / Edge 0.97689 / LPIPS 0.04601`，已线上 `65.6887`；
  这是当前已确认线上 best。
- 上一版 HAT+Mamba 离线 best：
  当前 native-io clean HAT 8-TTA `alpha=0.90` + 官方 MambaIRv2-Large x2 非 TTA `alpha=0.10`，raw clean full120 `41.72977`。
- 上一版 HAT-HAT 离线 best：
  当前 native-io clean HAT `alpha=0.65` + 上一版非 native-io clean HAT `alpha=0.35`，8-TTA，`interp=cubic`、`blend_interp=0.015`
  clean full120 `41.72349`；router k3/k4 selected gain 为 `0`，暂不生成新包。
- 最新平台口径重排：
  新增 `src/rank_platform_candidates.py`，用 `65.6887` 反馈重新拟合后，四指标 ridge 口径第一仍是已经提交的 fixed low-Mamba `0.60/0.35/0.05 raw`：
  `PSNR 33.87895 / SSIM 0.99797 / Edge 0.97689 / LPIPS 0.04601 / proxy 41.72854`，ridge 估计 `65.7003`，proxy-fit `65.7244`；
  未提交候选里，`0.65/0.30/0.05 raw` 接近但略低，ridge `65.6997`；k4 ridge router 估计 `65.6994`；HAT-HAT `alpha=0.65 + blend=0.005` 估计 `65.6960`；
  三路 raw 是 proxy best `41.73174`，proxy-fit 估计约 `65.7303`，但 ridge 因 LPIPS `0.04657` 只估约 `65.692`；
  这些都只是下一提交窗口的小刷/赌博候选，不是 `66` 强信号。
- 经验平台分反推：
  10 个合法提交拟合为 `platform ~= -11.246223 + 1.844556 * local_proxy`；
  估计线上 `66` 需要 local proxy `41.87795`，`67` 需要 `42.42009`；
  当前离线 best `41.73174` 距离估计 `66` 仍差约 `+0.146`。
- 固定 `probe40` 基线：`43.08219`
- 注意：`HAT-L official-only p96` 的 `41.62784/41.62860` 来自 `val_probe40_seed42` 训练后对 `full120` 的污染复核，不能当已确认 clean `full120`；原因是 `probe40` train 与 `full120` 有 `80/120` 训练重叠。
- 当前最好已确认 clean checkpoint：
  `checkpoints_hat_l_nativeio_official_cont_limit1680_lr8e8_full120clean_avg/best_x2.pth`
- 上一版 clean incumbent checkpoint：
  `checkpoints_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`
- 更早一版 clean baseline checkpoint：
  `checkpoints_hat_l_official_cont_lr3e6_from_ep4_p48_avg/best_x2.pth`
- 当前主线：
  普通 `native-io HAT-L` continuation 已冻结；最新 clean raw `full120` 已到 `41.63526`，blend002 到 `41.64162`
- 最新完成的 continuation：
  `checkpoints_hat_l_nativeio_official_cont_limit1680_lr8e8_full120clean_avg`
- 最新 quick ensemble 结论：
  `HAT-HAT` 最好只到 `proxy 41.63031`，现在已低于当前 clean best，不是提交候选
- 当前高赔率活动线：
  `trainall1800 + blend002` 已线上 `65.5414`，相对旧 legal best `65.4881` 为 `+0.0533`；
  clean-vs-trainall A/B 包 `submission/fqy_hat_l_nativeio_lr8e8_blend002.zip` 已线上 `65.5408`；
  clean 比 train-all 低 `0.0006`，且 `<65.56`，所以 HAT-family polishing/router/continuation 提交通道冻结；
  之后重新复核合法 8-TTA self-ensemble，clean `full120` 到 `41.72045`，已过 `41.66` 打包线；
  已生成并校验 `submission/fqy_hat_l_nativeio_clean_tta_blend002_cubic.zip`；
  当时离线续推的 TTA 后处理 sweep 最好 `41.72060`，HAT image-space ensemble 最好 `41.72349`，都不是 `66` 强信号；
  当前没有活跃训练/评估进程；若 `pgrep` 为空属正常
- 最新合法 router / metric-loss 复核：
  clean HAT 8-TTA full120 后处理 oracle `41.72830`，HAT-HAT 8-TTA full120 alpha/route oracle `41.73022`，只有 `+0.006~0.008` 上限，不足以支撑复杂 router 冲 `66`；
  metric-loss continuation 已关闭：
  `logs/train_x2_hat_l_nativeio_metricloss_limit512_lr5e8_full120clean_avg.log` 从 `41.63526` 掉到 `41.63219`；
  `logs/train_x2_hat_l_nativeio_ssimedge_probe40_limit512_lr5e8_avg.log` 从 `43.09584` 掉到 `43.09211`。
- 当前新开的自研模型线：
  `ThermalEdgeSR` 已完成代码接入，也补了 HAT-teacher 蒸馏支持；
  但 `tiny from-scratch` 和 `small + HAT distillation` 两轮官方 `probe40` 都失败，日志是
  `logs/train_x2_thermal_edge_tiny_probe40_limit512_p96.log`
  `logs/train_x2_thermal_edge_small_kdhat_probe40_limit512_p96.log`
- 当前记录修复：
  `experiments/probe_summary_20260510.csv` 的 `0.01` 阈值已按“相对 baseline 的增量阈值”统一解释，旧误标记录已修正
- 当前公开强权重复核：
  `DAT_X2` Stage A smoke4：`44.52328`，低于门槛 `45.20` 和 HAT 同切片约 `45.39403`，关闭；
  `DRCT x2` 暂无可复现公开 x2 checkpoint，关闭；
  `FGA-SR` 当前公开权重为 x4，没有 x2 eval-only 候选；
  `SRFormer/SRFormerV2` 复用 2026-05-09 smoke 失败结论，不回开；
  `ONNX/OpenModelDB/HF` 最新 Stage A 最好仍是 HAT-SRx2 ONNX `44.76882`，其后是 SwinIR classical DF2K `44.48555`、Xenova Swin2SR classical x2 ONNX `44.41876`、Xenova Swin2SR lightweight x2 ONNX `44.13388`、SwinIR classical DIV2K `44.10949`、SwinIR lightweight `43.96717`、SRFormerLight `43.95802`、SPAN x2 ch48 `43.75943`，均低于 `45.20`，关闭；
  2026-05-11 追加补扫也未过门槛：
  HF `jaideepsingh/upscale_models` 标准 HAT x2 `44.66887`；
  OmniSR DF2K/DIV2K `43.72507 / 43.60434`；
  GRL tiny/small/base `43.57378 / 43.88553 / 44.28306`；
  TorchSR NinaSR-B2/RCAN/EDSR `43.81458 / 43.87868 / 43.87123`；
  SwinFIR-T/SwinFIR/HATFIR `44.17709 / 44.72439 / 44.94415`；
  HAT 8-TTA + HATFIR external ensemble best `45.49817`，仍低于 HAT-HAT smoke best `45.51113`；
  HATFIR official x2 full120 `41.35036`，HAT TTA vs HATFIR full120 oracle 只有 `41.72650` 且 k3/k4 LR-cluster selected gain 为 `0.00000`，关闭；
  冷门 OpenModelDB Swift-SRGAN x2 `42.20725`、realSR SwinIR GAN x2 `38.83051`，均低于 Stage A；`2x-PSNR` Google Drive 链接当前无法被 `gdown` 获取；
  MAN tiny/light/base `43.64204 / 44.03555 / 44.28076`；
  ThalisAI Swin2SR x2、AMD RCAN/SESR ONNX、RealESRGAN/BSRGAN/MoSR/RealPLKSR/DITN/Compact 系列均未接近门槛；
  Qualcomm QuickSRNet Small/Medium/Large metadata 是 x4，非本赛 x2，排除。
- 2026-05-11 最新追加合法探针：
  x4 内部输出再确定性下采样到 x2：`DAT_2_x4` best `44.63012`、`DAT_2_x3` `42.51445`、`SwinIR DF2K x4` best `44.42676`，均低于 Stage A `45.20`，关闭；
  HF 标准 HAT x2 单模型 `44.66887`，HAT+HF HAT ensemble best `45.49777`，仍低于 HAT-HAT `45.51113`；
  OpenModelDB `DAT_2_x2` `44.53943`，关闭；
  HAT SWA 已新增 `src/average_checkpoints.py`，当前+lr15e7 50/50 `45.49383`、当前+clean1024 50/50 `45.49424`，均低于当前 TTA 基线；
  HAT refiner 已新增 `src/hat_refiner.py` / `src/train_hat_refiner_gray.py` / `src/evaluate_hat_refiner_gray.py`，512 patch probe 后 no-TTA smoke4 从 `45.38694` 掉到 `45.38152`，关闭；
  `src/evaluate_hat_gray.py` 已新增 `--tile-size/--tile-overlap`，`tile96 overlap24` no-TTA `45.33157`，关闭；
  `src/rank_platform_candidates.py` 已新增；`65.6887` 反馈后平台 ridge 口径仍把已提交 fixed low-Mamba 排第一，未提交候选预计也只有 `65.69~65.70` 左右；
  官方 MambaIRv2 x2 已重新拉取并评估：单支 Base/Large smoke4 `44.54960/44.55586` 关闭；HAT 8-TTA + MambaIRv2-Large 非 TTA 过 Stage B，probe40 `43.20532`，full120 raw `41.72977`；LR-only blend smoke 下降，raw 最好；
  MambaIRv2-Large official-only 微调单支 full120 `41.02053 -> 41.27405`，但 HAT+微调 Mamba full120 只有 `41.72732`，低于原版 Mamba ensemble，关闭；
  新增 `src/evaluate_hat_hat_mambair_ensemble.py`；current HAT 8-TTA + previous HAT 8-TTA + MambaIRv2-Large 非 TTA 三路 ensemble：smoke4 `45.51503`，probe40 `43.20913`，full120 `0.60/0.30/0.10 raw = 41.73174`，刷新 proxy best；三路 LR-only blend smoke 下降，raw 最好；HAT-B 换 native-io clean1024 的三路 smoke4 只有 `45.50314`，关闭；
  2026-05-12 追加三路 Mamba-TTA sanity：固定 smoke4 最好 `0.55/0.35/0.10 raw = 45.51498`，略低于三路 Mamba non-TTA `45.51503`，关闭，不进 probe40/full120；
  2026-05-12 追加官方 clean HAT-L 大 patch 探针：`p128` 与 `p112` 都在初始 probe40 `43.09584` 后第一步训练 CUDA OOM，关闭，不再硬推大 patch；
  2026-05-12 追加 CAT-R/CAT-R2 x2 smoke4：`44.50241 / 44.50670`，低于 Stage A `45.20`，关闭；
  2026-05-12 追加三路低 Mamba fine smoke4：最好 `0.55/0.375/0.075 raw = 45.51573`，只比旧三路 smoke best `45.51503` 高 `+0.00070`，同族 full120/ridge 仍不如已提交 fixed low-Mamba，暂不进 probe40/full120；
  2026-05-12 追加 NTIRE2026 infrared x4 官方仓库补扫：`external_models/NTIRE2026_infraredSR` 已 clone，按合法 x4 输出再 `area` 下采样到 x2 做 Stage A；team00 DAT `44.50308`，team03 `hat_1` `40.00006` 且四模型 HAT/PFT ensemble 单图 `240s` timeout，team10 HAT-L `40.18305 / 40.04407`，team09 MambaIRv2 `40.11812`，均低于 `45.20`，关闭；
  A2D2 visible-road 旧线复核：full static 最优 total score 约 `0.43984`，高于之前 visible-road probe gate `0.43`，不重开 matched-data/probe；
  新增 `src/make_hat_hat_mambair_routed_submission.py`，支持 fixed default route 或 cluster routed HAT/HAT/MambaIRv2 合法包；目前只补能力，未生成 test PNG/zip；
  HAT-HAT/HAT+Mamba/三路 route pool 的 k4 ridge router：full120 `41.72627`，updated ridge `65.6994`；
  low-Mamba full120：`0.60/0.35/0.05 raw = 41.72854`，updated ridge `65.7003`，已提交并返回 `65.6887`，仍是当前平台拟合第一；
  `CallMeDaniel/NTIRE2026-InfraredSR` RFRSR x4 权重已下载但官方推理仓库当前无法匿名 clone、权重含自定义 PDA/DCN，缺源码不稳妥复现；`Kronbii/thermal-super-resolution` 已 clone 但无 `.pth/.pt/.onnx` 权重。
- 当前 LR-only sanity：
  `src/evaluate_hat_ttt_gray.py` 已支持 native-io TTT；
  当前 HAT smoke4 raw `45.38694`，`blend002 + cubic` `45.39291`，TTT 最好 `45.39207`，都只是 polishing，不进提交。
- 当前 8-TTA 候选：
  smoke4：train-all raw `45.38967`，train-all `TTA+blend002+cubic 45.49316`，clean `TTA+blend002+cubic 45.49347`；
  clean full120：`PSNR 33.87159 / SSIM 0.99795 / Edge 0.97687 / LPIPS 0.04628 / proxy 41.72045`；
  包校验：`101` entries、`100` PNG、单一顶层目录、全 PNG 为 `640x512` 8-bit grayscale。
- 当前总策略：
  当前不再提交普通小修；先离线围绕 `8-TTA`、LR cluster router、clean/train-all TTA 对比和新公开 x2 权重继续找 `66` 级证据，等提交位恢复后再决定下一包。

要求：

- 只走合法路线：纯模型、公开数据训练、合法后处理、合法 ensemble
- 不用训练 HR 直接替换测试输出
- 不做 retrieval/reference 灰区包
- 生成提交包时必须保持平台兼容层级：zip 内只能有一个顶层 team 目录，目录内放 `README.md` 与 `preliminary/*.png`；不要把 `README.md` 和 `preliminary/` 直接放在 zip 根目录
- 没有明显强信号前，不生成 test PNG/zip，不浪费提交次数
- 优先延续当前最顺的合法主线；不要回头重开已经关闭的数据源分支

请先给我一句当前状态判断，然后直接继续。
