# 睿创微纳 AI 新睿人才“星探计划”技术说明

## 1. 方案概述
团队名称：fqy

本方案当前第一阶段合规主线采用 HAT-L 公开预训练模型迁移，在官方训练数据上做低学习率微调与合法 continuation，并只使用纯模型推理、公开预训练权重、官方训练数据微调、合法后处理与合法模型融合；不使用训练集 HR 直接替换测试输出。RCAN / SwinIR 等旧路线仅作为历史归档保留。

## 2. 训练策略
- x2 第一阶段：优先使用 HAT-L x2 公开预训练权重，在官方训练集上进行合法 `official-only` 微调与 continuation；当前项目内最强 clean 本地主线已升级到 `native-io HAT-L` continuation，`p48/p96` 低学习率微调为其前序基线。
- x4 第二阶段：当前以历史 RCAN 方案为归档，不作为本轮主线。
- 损失函数：当前合规主线以像素损失为主；感知项和复杂混合 loss 仅作探针，不作为稳定提交主线。
- 数据增强：随机裁剪、水平/垂直翻转、90 度旋转。

## 3. 复现命令
当前第一阶段线上最好：
- 平台分：65.4881
- 提交包：`submission/fqy_hat_l_ft_ep4_pure_rootdir.zip`
- 权重：`checkpoints_hat_l_official_full_lr1e5_p48_avg/best_x2.pth`
- 架构：HAT-L x2 ImageNet 公开预训练模型 + 官方训练集低学习率微调 epoch4
- 推理：灰度输入重复为 RGB，模型输出转灰度，no-TTA，无 sharpen，无训练 HR 替换
- 本地验证：PSNR 33.766150，SSIM 0.997900，Edge 0.976230，LPIPS 0.046190，proxy 41.613630

当前项目内已确认的更强 clean 本地 checkpoint：
- 权重：`checkpoints_hat_l_nativeio_official_cont_limit1680_lr15e7_full120clean_avg/best_x2.pth`
- 本地验证：PSNR 33.78392，SSIM 0.99789，Edge 0.97648，LPIPS 0.04503，proxy 41.63416
- 说明：这是当前 clean `full120` best，但尚未打包提交，因为相对已提交线上 best 的本地涨幅仍偏小，预计仍主要是 `65.5x` 档小涨。

上一版 HAT-L pure：
- 平台分：65.1093
- 提交包：`submission/fqy_hat_l_pure_rootdir.zip`
- 权重：`external_models/HAT_weights/HAT-L_SRx2_ImageNet-pretrain.pth`
- 本地验证：PSNR 33.511440，SSIM 0.997860，Edge 0.975050，LPIPS 0.049220，proxy 41.350250

当前合法冲分方向：
- HAT-L 官方训练集低学习率微调与 continuation。
- `native-io HAT-L`、合法轻量后处理、合法 HAT-family ensemble。
- 公开数据预训练后回拉官方训练集微调，但没有强信号前不扩训练。

灰区留档，不作为合规主线：
- `submission/fqy_hat_l_nearest_thr40.zip`、`submission/fqy_nearest_hybrid_ens_thr40.zip` 使用训练 HR 近邻替换测试输出，只保留实验记录。

上一版线上最好：
- 平台分：62.508
- 权重：`checkpoints_pixel_lr1e5/best_x2_fp16_ep40_proxy3995579.pth`
- 架构：base RCAN
- 推理：8-TTA，`sharpen_amount=0.05`
- 本地验证：PSNR 32.2030，SSIM 0.99730，Edge 0.96310，LPIPS 0.05823，proxy 39.99651

上一版线上提交：
- 平台分：62.2878
- 权重：`checkpoints_pixel/best_x2_fp16.pth`
- 推理：8-TTA，`sharpen_amount=0.05`
- 本地验证：PSNR 32.0845，SSIM 0.99726，Edge 0.96195，LPIPS 0.05967，proxy 39.87263

上一版线上基线：
- 平台分：62.283
- 推理：8-TTA，无 blend，无 sharpen
- 本地验证：PSNR 32.0842，SSIM 0.99728，Edge 0.96185，LPIPS 0.06179，proxy 39.86799

上一版 large RCAN 提交包复现：
```bash
python3 src/make_submission.py \
  --phase 1 \
  --team-name fqy_rcan_large_ep60_tta_sharp005 \
  --weights-x2 checkpoints_large_pixel/best_x2_fp16.pth \
  --preset large \
  --tta \
  --sharpen-amount 0.05
```

备选 ensemble 候选：
- 提交包：`submission/fqy_rcan_large07_swinir03_ep32_tta_sharp005.zip`
- 权重组合：`0.7 * checkpoints_large_pixel/best_x2_fp16.pth + 0.3 * checkpoints_swinir_base_pixel_lr5e5_from_ep4/best_x2_fp16.pth`
- 推理：8-TTA，`sharpen_amount=0.05`
- 本地验证：PSNR 33.3664，SSIM 0.99771，Edge 0.97395，LPIPS 0.04804，proxy 41.20455
- 说明：当前本地最强候选，比 large 单模型精确验证高 `+0.04325`。

- 提交包：`submission/fqy_rcan_large09_base01_ep60_tta_sharp005.zip`
- 权重组合：`0.9 * checkpoints_large_pixel/best_x2_fp16.pth + 0.1 * checkpoints_pixel_lr1e5/best_x2_fp16_ep40_proxy3995579.pth`
- 推理：8-TTA，`sharpen_amount=0.05`
- 本地验证：PSNR 33.3280，SSIM 0.99771，Edge 0.97361，LPIPS 0.04826，proxy 41.16494
- 说明：本地只比 large 单模型高 `+0.00364`，适合作为第二提交候选。

复现命令：
```bash
python3 src/make_submission.py \
  --phase 1 \
  --team-name fqy_rcan_large_ep60_tta_sharp005 \
  --weights-x2 checkpoints_large_pixel/best_x2_fp16.pth \
  --preset large \
  --tta \
  --sharpen-amount 0.05
```

备选 ensemble 复现命令：
```bash
python3 src/make_submission.py \
  --phase 1 \
  --team-name fqy_rcan_large07_swinir03_ep32_tta_sharp005 \
  --weights-x2 checkpoints_large_pixel/best_x2_fp16.pth checkpoints_swinir_base_pixel_lr5e5_from_ep4/best_x2_fp16.pth \
  --ensemble-coeffs 0.7,0.3 \
  --preset auto \
  --tta \
  --sharpen-amount 0.05
```

```bash
python3 src/make_submission.py \
  --phase 1 \
  --team-name fqy_rcan_large09_base01_ep60_tta_sharp005 \
  --weights-x2 checkpoints_large_pixel/best_x2_fp16.pth checkpoints_pixel_lr1e5/best_x2_fp16_ep40_proxy3995579.pth \
  --ensemble-coeffs 0.9,0.1 \
  --preset auto \
  --tta \
  --sharpen-amount 0.05
```

下一步低成本冲第一名：
```bash
python3 src/train.py \
  --scale 2 \
  --preset large \
  --patch-size 96 \
  --batch-size 2 \
  --epochs 16 \
  --repeat 4 \
  --workers 4 \
  --val-count 120 \
  --val-every 4 \
  --save-every-val \
  --train-all \
  --lr 3e-6 \
  --min-lr 5e-7 \
  --ssim-weight 0 \
  --edge-weight 0 \
  --resume checkpoints_large_pixel/best_x2.pth \
  --out-dir checkpoints_large_trainall_lr3e6
```

SwinIR / Transformer 主线训练：
```bash
python3 src/train.py \
  --scale 2 \
  --preset swinir_base \
  --patch-size 96 \
  --batch-size 2 \
  --epochs 80 \
  --repeat 4 \
  --workers 4 \
  --val-count 120 \
  --val-every 4 \
  --save-every-val \
  --lr 2e-4 \
  --min-lr 1e-6 \
  --ssim-weight 0 \
  --edge-weight 0 \
  --out-dir checkpoints_swinir_base_pixel
```

继续 x2 纯像素微调：
```bash
python3 src/train.py \
  --scale 2 \
  --preset base \
  --patch-size 96 \
  --batch-size 8 \
  --epochs 40 \
  --repeat 4 \
  --workers 4 \
  --val-count 120 \
  --val-every 2 \
  --lr 1e-5 \
  --ssim-weight 0 \
  --edge-weight 0 \
  --resume checkpoints_pixel/best_x2.pth \
  --out-dir checkpoints_pixel_lr1e5
```

训练 x4：
```bash
python3 src/train.py --scale 4 --preset small --patch-size 64 --batch-size 6 --epochs 160 --repeat 4 --out-dir checkpoints_x4
```

生成第一阶段提交包：
```bash
python3 src/make_submission.py --phase 1 --weights-x2 checkpoints/best_x2_fp16.pth --tta
```

## 4. 环境说明
- Python 3.12
- PyTorch 2.9.1+cu129
- 核心依赖：torch, torchvision, opencv-python-headless, numpy, piq, lpips, tqdm
