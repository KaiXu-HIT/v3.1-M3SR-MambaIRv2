# GTSS-MambaSR v3.1：训练、测试与三模型对比

本版本依据 [用户提供的 v3.1 方案](GTSS_MambaSR_SPEC.md)，从 v3.0 提交 `697f97a` 修改。
GitHub 仓库：`https://github.com/KaiXu-HIT/v3.1-M3SR-MambaIRv2.git`。
本地项目目录继续为 `D:\Code\Python\M3SR-MambaIRv2`。

## 本次结构修改

- 活跃模型改为 `GTSSMambaIRv2`，删除其 DGE、RAGA、GCR 和 geometry_route_bias 路径。
- 保持 RGB route、hard Gumbel policy、prompt、排序、B/C projection、selective-scan kernel 和重建头的原有公式。
- 保留 Sobel 与 per-image P2/P98 edge normalization；GRE 缩小为 `4→16→8→1`，中间 GELU、输出 Sigmoid。
- 在 ASSB 2/4/6 的每个 ASSM 内，使用该 ASSM 实际的 `x_sort_indices` 同步 Gather depth 与 confidence。
- 路径跳变：`Q = clamp(abs(D_s[t]-D_s[t-1]), 0, 0.25)/0.25`。
- 端点可靠性：`R = min(C_s[t], C_s[t-1])`；`G = R*Q`，首 token 的 G 严格为 0。
- 每个被调制 ASSM 独立一个标量：`beta = 0.5*sigmoid(beta_logit)`，初始精确取 `log(0.02/0.48)`，使 beta≈0.02，共 18 个。
- 将 `[B,1,L]` gate 扩展到 `[B,1,1,L]`，对 `[B,K,d_inner,L]` 的 raw dts 加 `beta*G`，之后原样调用 `delta_bias` 与 `delta_softplus=True`。
- transition 必须逐 ASSM 在排序后生成，不能共享一张预计算的二维 transition map。RGB Sobel/depth Sobel 只用于 GRE。

GRE 含 1,761 个参数，加上 18 个 beta_logit，总新增 **1,779** 个参数。
RGB baseline：23,050,713；GTSS：23,052,492；增加约 **0.00772%**。

所有 DIV2K/五测试集的路径、filename templates、Depth 读取、整图 P2/P98 normalization、同步 crop/flip/rotation 保持不变。
GTSS 从头训练 500k，GT192/LR48、batch=2、单 GPU、seed10、L1、Adam lr=2e-4、betas=[0.9,0.99]，原 scheduler 不变。
不新增其他损失、不进行变体快筛，也不从 HR 生成 Depth。

日志包含 `l_pix`、confidence_mean/std、`beta_2_1` 到 `beta_6_6`（仅 2/4/6 阶段）、每个 ASSM 的 `gate_mean_*` 和 transition_gate_mean。

## 更新服务器代码

若服务器已经存在 v3.0 项目，目录无需改名。进入旧目录，更新 remote 后拉取：

```bash
cd /home/BRAIN/xukai/code/v3.0-M3SR-MambaIRv2
git remote set-url origin https://github.com/KaiXu-HIT/v3.1-M3SR-MambaIRv2.git
git pull --ff-only origin main
```

如果首次下载：

```bash
cd /home/BRAIN/xukai/code
git clone https://github.com/KaiXu-HIT/v3.1-M3SR-MambaIRv2.git
cd v3.1-M3SR-MambaIRv2
```

以下命令均在项目根目录、原 baseline 使用的 CUDA/Mamba Conda 环境下执行。
本地 Windows CPU 检查环境不替代服务器环境。

## 训练前检查

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gtss/check_gtss.py --check-data
```

默认使用真实 CUDA selective scan；检查完整模型前向、18 个 beta 的机制和梯度、baseline 等价性、checkpoint 往返。
`--check-data` 还检查全部样本的文件配对，并解码每个训练/验证/测试集的首、中、末样本。它不代表读取过所有图像或验证过 Depth 的生成来源。

## 从头训练

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_GTSS_MambaSR_x4.yml
```

实验目录：`experiments/v3.1_GTSS_MambaSR_x4/`。
不会读取 RGB/v3.0 权重进行 warm-start。新训练也不需要填写 v3.0 对比权重。

中断后恢复同一 GTSS 实验：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_GTSS_MambaSR_x4.yml --auto_resume
```

## 单独测试 GTSS

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_GTSS_MambaSR_x4.yml
```

默认加载 `experiments/v3.1_GTSS_MambaSR_x4/models/net_g_500000.pth`，严格加载全部权重。
保持 Set5、Set14、B100、Urban100、Manga109，原 partition/overlap、crop_border=4、Y-channel PSNR/SSIM。
如需指定其他已确定的权重文件：

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_GTSS_MambaSR_x4.yml --force_yml path:pretrain_network_g=/absolute/path/to/gtss_checkpoint.pth
```

## RGB baseline 与 GTSS：各三次

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gtss/evaluate_repeated.py --seeds 10 11 12
```

默认 RGB 权重仍为：
`/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth`。
可用 `--baseline-checkpoint PATH` 或 `--gtss-checkpoint PATH` 覆盖对应权重。

每一对模型使用同一个 seed，三个重复使用 10/11/12。每个 worker 独立进程，模型构建后、各数据集推理前重置 RNG。
两/三个模型统一文件排序、cuDNN 设置和分块/指标流程，不关闭原 Gumbel 随机路由。
输出 mean ± sample std（ddof=1）；应使用本轮复测的 RGB 均值计算增益，不把附件中的单次成绩充当本轮复测结果。

## 何时补充 v3.0 权重

**在 GTSS 训练完成、准备做 RGB/v3.0/GTSS 三模型比较时补充。**
不影响训练、断点恢复、GTSS 单独测试或 RGB/GTSS 两模型复测。

预留位置：
`options/test/mambairv2/test_GTSS_GRS_reference_x4.yml` 的 `path.pretrain_network_g`。
当前值为：
`__SET_V3_0_GRS_CHECKPOINT_BEFORE_THREE_MODEL_COMPARISON__`。

填写该字段后：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gtss/evaluate_repeated.py --include-grs --seeds 10 11 12
```

也可以不修改 YAML，在运行时提供完整路径：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/gtss/evaluate_repeated.py --include-grs --grs-checkpoint "/absolute/path/to/v3.0/net_g_500000.pth" --seeds 10 11 12
```

路径未填写时，三模型命令会在开始任何推理前报出明确提示，不会静默跳过 GRS，也不会用 GTSS 权重代替它。
旧版 `GRSMambaIRv2` 及其 backbone 已隔离用于历史权重评测：`basicsr/archs/legacy_grs/`。
其 state_dict 名称和前向计算兼容 v3.0；它不被 GTSS 实例化。
旧训练选项和说明保留为历史记录，新的默认入口均为 GTSS。

## 输出与判断

结果位于 `results/gtss_comparison/summary.md` 与 `summary.json`；含各 seed 原始结果、各模型 PSNR/SSIM mean/std 和相对 RGB 的差值。
三模型模式共运行九个推理任务；默认两模型模式共六个。
按照新方案记录：五库平均差是否 >0、是否 >=0.05 dB、Urban100 是否 >=0.08 dB、Manga109 是否 >=0.10 dB，以及 B100 的实际差值。
不再沿用 v3.0 的 +0.03 dB / 4-of-5 成功判定。
文档中的停止规则需要结合完整训练的真实结果判断，不由合成检查数据推断。

## 本地检查范围

```powershell
.\.venv\Scripts\python.exe scripts/gtss/check_gtss.py --cpu-reference
```

CPU 模式执行实际源码与显式可微 selective-scan recurrence，以检查公式、梯度、shape、数据同步和 baseline 等价性。
不代表 CUDA extension 集成、服务器数据可访问性或真实 PSNR 提升；详见 [本次验证记录](GTSS_VERIFICATION.md)。
