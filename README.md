# Rcwn-Competition

Stage-1 x2 图像超分辨率竞赛工作区。仓库记录了从模型评估、TTA、融合、提交打包到线上分数追踪的一整套实验流程，主要用于继续冲击线上分数和保留可复盘的竞赛现场。

## 当前状态

最新交接信息见 [HANDOFF.md](./HANDOFF.md)。截至当前仓库记录：

- 已确认的最高合法线上分数为 `65.9109`。
- 对应提交文件名记录为 `submission/fqy_hat_hat_ssttta_275225500_raw.zip`。
- 该提交融合了当前 HAT 8-TTA、上一版 HAT 8-TTA 和 SSTXLarge_Plus_DFLIP_X2 8-TTA。
- 当前目标是继续尝试冲击 `66` 分。
- 没有记录正在运行的训练或评估进程。

HAT 系列普通续训已经冻结。新候选建议先通过离线门槛再进入大规模提交流程：`full120 >= 41.66` 或 `probe40 >= 43.12`。

## 仓库内容

- `HANDOFF.md`：最新交接说明和下一步策略。
- `PROJECT_LOG.md`：实验过程记录。
- `SCORING_NORMALIZATION.md`：线上/离线分数归一化说明。
- `SERVER_MIGRATION_20260513.md`：服务器迁移记录。
- `experiments/`：实验 CSV 和分数记录。
- `manifests/`：数据、模型、提交清单。
- `src/`：训练、评估、融合、提交相关脚本。
- `rcwn.md`：竞赛背景和任务说明。

数据集、外部模型、checkpoint、日志和 submission zip 通常体积较大，不应默认视为 Git 跟踪资产。继续实验前请先对照 `manifests/` 和 `HANDOFF.md` 检查本地路径。

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要依赖包括 `torch`、`torchvision`、`opencv-python-headless`、`timm`、`einops`、`basicsr`、`piq`、`lpips` 和 `tqdm`。

## 推荐工作流

1. 先阅读 [HANDOFF.md](./HANDOFF.md)，确认当前最好提交、冻结策略和下一步候选。
2. 用 `manifests/` 检查本地数据、模型和提交文件是否齐全。
3. 先跑 probe 或 full120 离线评估，不要直接扩大训练。
4. 只有离线结果过门槛时，再进入 TTA、融合和提交打包。
5. 新结果写回 `experiments/` 和 `PROJECT_LOG.md`，避免分数口径混乱。

## 注意事项

- README 中不再使用旧的绝对服务器路径；需要路径时请以本仓库相对路径或 `HANDOFF.md` 为准。
- 线上分数与离线 proxy 并非线性等价，融合策略必须结合 `SCORING_NORMALIZATION.md` 阅读。
- 本仓库面向竞赛复盘和继续实验，不包含完整数据闭环。

## 许可证

当前仓库未包含独立 `LICENSE` 文件。如需公开复用或分发，请先补充明确的开源许可证。
