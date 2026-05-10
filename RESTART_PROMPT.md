# 新对话启动提示词

我在 `/home/lkc/lkcproject/rcwn` 做 RCWN 比赛第一阶段 `x2` 合法冲分，目标是在严格遵守 `rcwn.md` 规则的前提下冲到线上 `70+`。请先读取：

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

- 当前最好已确认合法线上分：`65.4881`
- 当前最好已确认本地 clean `full120`：`41.63416`
- 固定 `probe40` 基线：`43.08219`
- 注意：`HAT-L official-only p96` 的 `41.62784/41.62860` 来自 `val_probe40_seed42` 训练后对 `full120` 的污染复核，不能当已确认 clean `full120`；原因是 `probe40` train 与 `full120` 有 `80/120` 训练重叠。
- 当前最好已确认 clean checkpoint：
  `checkpoints_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`
- 上一版 clean incumbent checkpoint：
  `checkpoints_hat_l_official_p96_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`
- 更早一版 clean baseline checkpoint：
  `checkpoints_hat_l_official_cont_lr3e6_from_ep4_p48_avg/best_x2.pth`
- 当前主线：
  `native-io HAT-L` continuation；最新 clean `full120` 已到 `41.63416`
- 最新完成的 continuation：
  `checkpoints_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg`
- 最新 quick ensemble 结论：
  `HAT-HAT` 最好只到 `proxy 41.63031`，现在已低于当前 clean best，不是提交候选
- 当前高赔率活动线：
  `native-io HAT-L` 已完成代码接入；
  `probe40` 首轮结果 `43.08025 -> 43.09142`，相对固定基线 `43.08219` 为 `+0.00923`
  `full120 clean 1024` 结果 `41.62143 -> 41.62969`
  `full120 clean 1680` 结果 `41.62969 -> 41.63416`，相对上一版 incumbent clean best `41.63025` 为 `+0.00391`
  当前没有活跃训练/评估进程；若 `pgrep` 为空属正常
- 当前新开的自研模型线：
  `ThermalEdgeSR` 已完成代码接入，也补了 HAT-teacher 蒸馏支持；
  但 `tiny from-scratch` 和 `small + HAT distillation` 两轮官方 `probe40` 都失败，日志是
  `logs/train_x2_thermal_edge_tiny_probe40_limit512_p96.log`
  `logs/train_x2_thermal_edge_small_kdhat_probe40_limit512_p96.log`
- 当前记录修复：
  `experiments/probe_summary_20260510.csv` 的 `0.01` 阈值已按“相对 baseline 的增量阈值”统一解释，旧误标记录已修正
- 当前总策略：
  保留一条低风险合法慢爬主线做保底，但把“找到结构性大跳模型/训练信号”作为冲 `70+` 的主目标；普通 `+0.00x` 本地涨幅不打包、不提交。

要求：

- 只走合法路线：纯模型、公开数据训练、合法后处理、合法 ensemble
- 不用训练 HR 直接替换测试输出
- 不做 retrieval/reference 灰区包
- 没有明显强信号前，不生成 test PNG/zip，不浪费提交次数
- 优先延续当前最顺的合法主线；不要回头重开已经关闭的数据源分支

请先给我一句当前状态判断，然后直接继续。
