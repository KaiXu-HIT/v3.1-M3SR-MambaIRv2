# GRS-MambaSR：面向 RGB ×4 超分的深度几何可靠性引导 MambaIRv2 方案

> **GRS-MambaSR = Geometry-Reliability Selective Mamba Super-Resolution**  
> 基线：`KaiXu-HIT/v1.0-M3SR-MambaIRv2`  
> 主任务：RGB ×4 classical SR  
> 主模态：LR RGB  
> 辅助模态：与 LR RGB 对齐的 Depth / pseudo-depth  
> 核心目标：在不牺牲原 MambaIRv2 RGB 重建能力的前提下，利用 Depth 提供的几何边界与区域结构信息，使五个经典测试集上的 PSNR/SSIM **稳定超过当前 RGB baseline**。

当前 baseline：

- Set5：32.7645 / 0.9029
- Set14：29.0786 / 0.7928
- B100：27.8639 / 0.7468
- Urban100：27.3198 / 0.8215
- Manga109：31.8458 / 0.9239

---

# 1. 方案结论与边界

本方案不采用“再堆一个 cross-attention / concat / 第二套 Mamba”这种拼接式思路，而是围绕一个更具体的问题设计：

> **Depth 对 RGB SR 最有价值的是几何结构，但 pseudo-depth 可能含噪、错边界或仅重复 LR RGB 已有信息，因此 Depth 不能与 RGB 平权融合，也不能无条件注入。**

模型遵循三个原则：

1. **RGB 始终是唯一重建主干（fidelity anchor）**；
2. **Depth 只作为 geometry prior，以残差方式提供补充，不替代 RGB 特征**；
3. **Depth 不仅影响 feature，还用于约束 MambaIRv2 的 token/state routing，但必须受到 reliability gate 控制。**

任何人在训练前都无法科学保证“90% 概率一定超过 baseline”。本方案能做到的是排除此前已暴露出的高风险设计（early concat、对称双流、无条件全局融合、文本 FiLM），并采用更符合 guided SR 文献与 MambaIRv2 结构特性的设计，形成一个值得一次完整训练验证的高优先级模型。

如果该模型完整训练后仍不能超过 baseline，应认真重新评估当前 pseudo-depth 对 RGB ×4 SR 的信息价值，而不是继续无限叠加模块。

---

# 2. 为什么不再采用此前的 early concat

此前 RGB+Depth 方案：

```text
RGB   -> Conv -> F_rgb ----\
                            concat -> 1x1 Conv -> MambaIRv2
Depth -> Conv -> F_depth --/
```

其问题是：

\[
F_{rgb}
\]

被新的随机融合结果

\[
F_{fuse}=Conv_{1\times1}([F_{rgb},F_d])
\]

替换，这会在 Depth 尚未证明有效时先改变整个 MambaIRv2 后续 backbone 的输入分布。

新的 GRS-MambaSR 改为：

\[
F_i^{out}=F_i+\alpha_i\cdot C_i\odot A_i\odot R_i^D
\]

其中：

- \(F_i\)：RGB 主干第 \(i\) 阶段特征；
- \(R_i^D\)：Depth 几何残差；
- \(C_i\)：空间可靠性图；
- \(A_i\)：通道选择门；
- \(\alpha_i\)：小幅可学习残差尺度。

Depth 只能“补充”，不能“覆盖”原 RGB 表征。

---

# 3. 公开研究给出的关键启示

## 3.1 非对称模态角色

Guided image SR 近期工作指出 target 与 guidance 不应被简单对称处理，不同模态应承担不同角色；不同层级的低层细节与高层结构也不应使用同一种融合方式。

因此本课题采用：

```text
RGB   = target / reconstruction stream
Depth = geometry guidance stream
```

而不是：

```text
RGB <-> Depth
```

平权双流。

## 3.2 需要 selective guidance

D2A2、DORNet、SigNet 等工作都强调跨模态误对齐、错误边界、纹理过迁移和“并非所有 guidance 都可信”的问题。因此 Depth 注入前必须估计当前位置的几何可信度，而不是直接 concat。

## 3.3 Depth 最有价值的是结构与边界

大量 guided depth SR 工作利用 depth/RGB gradient、高频、边界或几何结构进行选择性融合。对 RGB SR 来说，Depth 更适合帮助：

- 前景/背景边界；
- 物体轮廓；
- 几何分区；
- 重复结构中的层次关系；
- 高频恢复位置选择。

## 3.4 MambaIRv2 的 routing 是更有价值的切入点

当前 baseline 的 ASSM 中，RGB token 先经过 `route(x)`，再经 `gumbel_softmax` 形成语义 routing policy，并据此排序后执行 selective scan。

因此 Depth 不一定应该大量“写入 RGB feature”；它更适合作为：

> **几何 routing bias**

让 Mamba 的状态传播路径更符合几何分区。

---

# 4. 模型总体结构

模型名称：

## GRS-MambaSR

全称：

**Geometry-Reliability Selective Mamba for RGB Image Super-Resolution**

整体：

```text
                     ┌─────────────────────────────┐
                     │        Depth Input          │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                         Depth Geometry Encoder
                                    │
                         ┌──────────┴───────────┐
                         │                      │
                         ▼                      ▼
                   Geometry Feature       Depth Gradient
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                     Geometry Reliability Estimator
                                    │
                               Confidence C
                                    │
        ┌───────────────────────────┼──────────────────────────┐
        │                           │                          │
        ▼                           ▼                          ▼
   Geo Adapter #1             Geo Adapter #2             Geo Adapter #3
        │                           │                          │
        │                           │                          │
RGB -> Shallow -> ASSB1 -> [G-ASSB2] -> ASSB3 -> [G-ASSB4] -> ASSB5 -> [G-ASSB6]
                            │                         │                    │
                            └ Geometry-Conditioned Routing ───────────────┘
                                                      │
                                                      ▼
                                             Reconstruction Head
                                                      │
                                                      ▼
                                                    SR RGB
```

`[G-ASSB]` = 原 ASSB + 两类 Depth guidance：

1. **RAGA：Reliability-Aware Geometry Adapter**
2. **GCR：Geometry-Conditioned Routing**

RGB backbone、upsampling head、原 selective scan 主体尽量保持不变。

---

# 5. 模块一：Depth Geometry Encoder（DGE）

## 5.1 输入

Depth：

\[
D\in \mathbb{R}^{B\times1\times H\times W}
\]

若现有 depth 为 pseudo-depth，则继续使用当前数据，不从 HR RGB 生成，避免信息泄漏。

Depth 采用 robust P2/P98 normalization：

\[
D_n=\operatorname{clip}
\left(
\frac{D-P_2(D)}{P_{98}(D)-P_2(D)+\epsilon},
0,1
\right)
\]

## 5.2 显式几何梯度

固定 Sobel：

\[
G_x=K_x*D_n,\qquad G_y=K_y*D_n
\]

\[
E_D=\sqrt{G_x^2+G_y^2+\epsilon}
\]

输入：

\[
X_D=[D_n,E_D]
\]

同时提供：

- relative depth；
- depth discontinuity；
- object boundary；
- region transition。

## 5.3 Encoder

```text
[D, E_D]        2 channels
    ↓
3×3 Conv        2 -> 48
    ↓
GELU
    ↓
Depthwise 3×3 + Pointwise 1×1
    ↓
GELU
    ↓
Depthwise 3×3 + Pointwise 1×1
    ↓
3×3 Conv        48 -> 174
    ↓
F_D
```

输出：

\[
F_D\in\mathbb{R}^{B\times174\times H\times W}
\]

不构造第二套大 Mamba/Transformer backbone，避免 Depth 变成第二主干。

---

# 6. 模块二：Geometry Reliability Estimator（GRE）

## 6.1 RGB 边缘

LR RGB 转 luminance：

\[
Y=0.299R+0.587G+0.114B
\]

Sobel：

\[
E_R=|\nabla Y|
\]

对 \(E_R,E_D\) 做 per-image robust normalization。

## 6.2 Reliability 输入

\[
Z=[E_R,E_D,|E_R-E_D|,E_R\odot E_D]
\]

分别表示：

- RGB 结构；
- Depth 几何边界；
- 跨模态冲突；
- 共同边界响应。

## 6.3 Reliability 网络

```text
Z (4 ch)
 ↓
3×3 Conv 4 -> 32
 ↓
GELU
 ↓
3×3 Conv 32 -> 16
 ↓
GELU
 ↓
1×1 Conv 16 -> 1
 ↓
Sigmoid
 ↓
C ∈ [0,1]^(H×W)
```

得到空间可靠性图：

\[
C(x,y)
\]

它不是硬 edge mask，而是学习“该位置是否值得使用 Depth”。

---

# 7. 模块三：Reliability-Aware Geometry Adapter（RAGA）

## 7.1 Depth residual

\[
R_i^D=P_i(F_D)
\]

其中 \(P_i\)：

```text
1×1 Conv 174 -> 174
GELU
DWConv 3×3
1×1 Conv 174 -> 174
```

## 7.2 RGB-conditioned channel gate

当前 RGB feature：

\[
F_i
\]

计算：

\[
z_i=GAP(F_i)
\]

再：

\[
A_i=\sigma(W_2\delta(W_1z_i))
\]

\[
A_i\in\mathbb{R}^{B\times174\times1\times1}
\]

由 RGB 当前 reconstruction state 决定哪些通道需要 Depth 补充。

## 7.3 最终 residual guidance

\[
\Delta F_i^D
=
\alpha_i\cdot C\odot A_i\odot R_i^D
\]

\[
F_i'=F_i+\Delta F_i^D
\]

## 7.4 初始化

建议：

\[
\alpha_i=0.05
\]

并将 \(P_i\) 最后一层初始化为小权重：

```text
std = 1e-3
bias = 0
```

目的：

- 初期 Depth 扰动很小；
- Depth branch 从第一个 iteration 就能收到梯度；
- 避免 early concat 重写 RGB feature。

---

# 8. 模块四：Geometry-Conditioned Routing（GCR）

这是与 MambaIRv2 最紧密结合的核心创新。

## 8.1 原 routing

\[
R_{rgb}=route(X)
\]

\[
P=GumbelSoftmax(R_{rgb})
\]

routing policy 决定：

- prompt；
- token 重排序；
- 后续 selective scan 路径。

## 8.2 几何 routing bias

Depth token：

\[
D_t=Flatten(F_D)
\]

Depth route：

\[
R_D=W_DD_t
\]

其中：

\[
R_D\in\mathbb{R}^{B\times HW\times N_{token}}
\]

可靠性 token：

\[
C_t=Flatten(C)
\]

最终：

\[
R'
=
R_{rgb}
+
\rho_i\cdot C_t\odot R_D
\]

再执行：

\[
P=GumbelSoftmax(R')
\]

建议：

\[
\rho_i=0.05
\]

`Depth Route Projection` 最后一层 small initialization。

Depth 不单独扫描，不构造第二套状态空间网络；它仅以受控方式影响 RGB token 的组织路径。

---

# 9. 为什么只在 ASSB 2 / 4 / 6 注入

Baseline：

```text
6 × ASSB
Depths = [6,6,6,6,6,6]
```

本方案：

```text
RGB shallow
  ↓
ASSB1
  ↓
G-RAGA + GCR @ ASSB2
  ↓
ASSB3
  ↓
G-RAGA + GCR @ ASSB4
  ↓
ASSB5
  ↓
G-RAGA + GCR @ ASSB6
  ↓
Reconstruction
```

理由：

1. 避免每层重复注入导致 Depth domination；
2. 三次渐进 guidance 覆盖 early/middle/deep 表征；
3. 符合 guided SR 中多阶段交互优于单次 early concat 的经验；
4. 参数与 FLOPs 增幅可控；
5. 不需要先做大规模 location search。

---

# 10. 完整前向流程

### Step 1：RGB 主干浅层特征

\[
F_0=Conv_{first}(I_{LR})
\]

### Step 2：Depth 几何编码

\[
E_D=|\nabla D|
\]

\[
F_D=DGE([D,E_D])
\]

### Step 3：可靠性估计

\[
E_R=|\nabla Y(I_{LR})|
\]

\[
C=GRE([E_R,E_D,|E_R-E_D|,E_RE_D])
\]

### Step 4：渐进 RGB reconstruction

\[
F_1=ASSB_1(F_0)
\]

\[
\hat F_1=RAGA_2(F_1,F_D,C)
\]

ASSB2 routing：

\[
route_2'
=
route_2(\hat F_1)
+
\rho_2C\odot route_D(F_D)
\]

随后：

\[
F_2=GASSB_2(\hat F_1,F_D,C)
\]

再依次：

\[
F_3=ASSB_3(F_2)
\]

\[
F_4=GASSB_4(RAGA_4(F_3,F_D,C))
\]

\[
F_5=ASSB_5(F_4)
\]

\[
F_6=GASSB_6(RAGA_6(F_5,F_D,C))
\]

### Step 5：重建头保持原 baseline

保留：

```text
conv_after_body
+
long residual
+
conv_before_upsample
+
pixelshuffle
+
conv_last
```

最终：

\[
I_{SR}=Recon(F_6)
\]

---

# 11. Loss：第一版只保留 L1

第一版完整模型只用：

\[
\mathcal{L}_{pix}=\|I_{SR}-I_{HR}\|_1
\]

不同时增加：

- perceptual loss；
- GAN loss；
- depth edge loss；
- CLIP loss；
- frequency loss。

原因：当前首先验证“架构 + Depth prior 是否能超过 RGB baseline”。同时改 loss 会使增益来源混杂。

---

# 12. 数据设置

保持 v1.0 baseline：

```text
Train: DIV2K
Scale: ×4
GT patch: 192
LR patch: 48
RGB degradation: bicubic
```

Depth：

```text
与 LR RGB 同 H×W
严格一一对应
与 RGB 同步 crop / flip / rotation
```

禁止：

```text
从 HR RGB 生成 test-time Depth
```

否则会引入额外 HR 信息。

---

# 13. 第一轮直接完整训练

本轮不先训练 Add / Concat / FiLM / Early / Late 等大量变体。

直接训练：

```text
GRS-MambaSR ×4
500k iterations
```

为了论文公平性，正式主实验建议**从头训练**，完全匹配 v1.0：

```yaml
optim_g:
  type: Adam
  lr: 2.0e-4
  weight_decay: 0
  betas: [0.9, 0.99]

scheduler:
  type: MultiStepLR
  milestones: [250000, 400000, 450000, 475000]
  gamma: 0.5

total_iter: 500000

pixel_opt:
  type: L1Loss
  loss_weight: 1.0
  reduction: mean
```

不建议第一版直接 warm-start 后额外训练，因为会造成 baseline 与新模型训练预算不一致。

---

# 14. 初始化策略

### DGE

正常 Kaiming / truncated normal。

### RAGA final projection

```text
normal(std=1e-3)
bias=0
```

### alpha

```text
0.05
```

### Depth routing projection

前层正常初始化，最后到 `num_tokens=128` 的 Linear：

```text
normal(std=1e-3)
bias=0
```

### rho

```text
0.05
```

目的：网络早期仍以 RGB SR 学习为主，Depth guidance 渐进进入。

---

# 15. MambaIRv2 Gumbel routing 的公平评测

当前 baseline 的 ASSM 使用：

```python
F.gumbel_softmax(pred_route, hard=True)
```

存在推理随机性。

本轮训练不修改该机制，以保持 baseline 定义一致。

最终只对两个模型：

```text
v1.0 baseline
GRS-MambaSR
```

做稳定性复测：

1. 相同 `manual_seed`；
2. 相同 test pipeline；
3. 每个最终模型重复 3 次 inference；
4. 报告 mean ± std。

这不是用于筛选结构，只是最终公平评估。

---

# 16. 参数与复杂度约束

新增参数控制：

\[
<1.5M
\]

最好相对 23M baseline：

\[
<5\%-7\%
\]

原则：如果 LR pseudo-depth 需要增加非常大的第二主干才能换来极小 PSNR 增益，则性价比不足。

---

# 17. 性能判定

当前 baseline：

| Dataset | PSNR / SSIM |
|---|---|
| Set5 | 32.7645 / 0.9029 |
| Set14 | 29.0786 / 0.7928 |
| B100 | 27.8639 / 0.7468 |
| Urban100 | 27.3198 / 0.8215 |
| Manga109 | 31.8458 / 0.9239 |

## 最低有效标准

\[
5\text{-dataset Avg PSNR} \ge B0+0.03dB
\]

且至少：

```text
4/5 数据集 PSNR 不低于 baseline
```

## 值得继续论文开发的标准

优先希望：

```text
Urban100    +0.10 dB 或以上
Manga109    +0.10 dB 或以上
B100        不下降
Set5/Set14  基本不下降或小幅提高
```

五数据集平均：

\[
+0.05\sim+0.10dB
\]

在强 baseline 上已值得继续。

## 强结果

如果：

```text
Urban100 / Manga109 +0.15~0.25 dB
5-set average >= +0.08 dB
```

且参数增幅 <7%，则该路线具有较强论文潜力。

---

# 18. 失败判定

完整 500k 后若：

\[
Avg\Delta PSNR\le0
\]

特别是：

```text
Urban100 <= baseline
Manga109 <= baseline
```

则不建议继续对当前 pseudo-depth 做更多：

- Cross-Mamba；
- attention；
- 更大 Depth backbone；
- 更多 fusion stage。

这时应优先得出：

> 当前 pseudo-depth 对该 DIV2K bicubic ×4 fidelity SR 的增量信息不足，或其误差超过潜在几何收益。

---

# 19. 论文创新点

如果模型有效，可形成三个相互关联的创新点。

## Innovation 1：Asymmetric Geometry Guidance

RGB 主重建、Depth 几何辅助：

\[
RGB=fidelity\ anchor
\]

\[
Depth=geometry\ prior
\]

避免 symmetric dual-stream / early fusion 对 RGB reconstruction space 的破坏。

## Innovation 2：Reliability-Aware Geometry Residual

显式建模：

\[
C=f(E_R,E_D,|E_R-E_D|,E_RE_D)
\]

再通过：

\[
\Delta F^D=\alpha C A R_D
\]

进行空间+通道双重选择性 residual guidance。

核心不是“加 attention”，而是建模 pseudo-depth 在 RGB SR 中“哪里可信、哪些 feature 应该被影响”。

## Innovation 3：Geometry-Conditioned State Routing

Depth 不做普通 concat，而作为：

\[
Mamba\ token/state\ routing\ prior
\]

通过：

\[
R'=R_{rgb}+\rho C R_D
\]

使 Mamba 长程状态传播受到几何关系约束。

---

# 20. 与已有工作的区别

### DCNAS

借鉴“target/guidance 不应完全对称、fusion location 很关键”的研究结论，但本方案不做 NAS。

### D2A2

借鉴“跨模态错位和 pseudo-depth 需要显式处理”的问题意识，但不使用其原 cross-modal aggregation 结构。

### DORNet / SigNet

借鉴 selective/degradation-aware guidance 理念，但本方案的 selector 是 RGB-depth geometry reliability，核心融合对象进一步进入 MambaIRv2 routing。

### MMSR

借鉴“空间模态比文本更适合作为 localization constraint”的思想，但本方案不是 diffusion，也不使用多 ControlNet 或 text conditioning。

### SNUM-Net / JIIF / PGSR / Deep Anisotropic Diffusion

这些工作证明 guided SR 中结构先验、渐进交互与显式几何约束具有价值；本方案只吸收其任务规律，不直接移植其网络结构。

---

# 21. 源码修改建议

基于：

```text
https://github.com/KaiXu-HIT/v1.0-M3SR-MambaIRv2.git
```

建议新增：

```text
basicsr/archs/grs_mambairv2_arch.py
basicsr/models/grs_mambairv2_model.py
basicsr/data/rgb_depth_paired_image_dataset.py
options/train/mambairv2/train_GRS_MambaSR_x4.yml
options/test/mambairv2/test_GRS_MambaSR_x4.yml
```

## 类结构

```text
DepthGeometryEncoder
GeometryReliabilityEstimator
ReliabilityAwareGeometryAdapter
GeometryConditionedASSM
GeometryConditionedAttentiveLayer
GRSMambaIRv2
```

---

# 22. 代码级接口建议

## DepthGeometryEncoder

```python
class DepthGeometryEncoder(nn.Module):
    def forward(self, depth):
        # robust normalized depth is provided by dataset
        edge = sobel_gradient(depth)
        feat = self.encoder(torch.cat([depth, edge], dim=1))
        return feat, edge
```

## GeometryReliabilityEstimator

```python
class GeometryReliabilityEstimator(nn.Module):
    def forward(self, rgb, depth_edge):
        rgb_y = rgb_to_luma(rgb)
        rgb_edge = sobel_gradient(rgb_y)
        z = torch.cat([
            rgb_edge,
            depth_edge,
            torch.abs(rgb_edge - depth_edge),
            rgb_edge * depth_edge
        ], dim=1)
        return torch.sigmoid(self.net(z))
```

## ReliabilityAwareGeometryAdapter

```python
class ReliabilityAwareGeometryAdapter(nn.Module):
    def forward(self, rgb_feat, depth_feat, confidence):
        depth_res = self.depth_proj(depth_feat)
        channel_gate = self.channel_gate(rgb_feat)
        residual = self.alpha * confidence * channel_gate * depth_res
        return rgb_feat + residual
```

## GeometryConditionedASSM

仅修改原 ASSM routing：

```python
rgb_route = self.route(x)
depth_route = self.depth_route(depth_token)
route = rgb_route + self.rho * confidence_token * depth_route
cls_policy = F.gumbel_softmax(route, hard=True, dim=-1)
```

其余 prompt / sort / selective scan / fold 保持原实现。

---

# 23. 训练日志额外记录

除：

```text
l_pix
PSNR
SSIM
```

额外记录：

```text
alpha_2, alpha_4, alpha_6
rho_2, rho_4, rho_6
mean(C)
std(C)
mean(|Depth residual|)
```

这些不是成功标准，只用于完整训练失败后定位：

- 是否完全忽略 Depth；
- Depth 注入是否过强；
- reliability 是否塌缩到 0/1；
- route bias 是否失控。

---

# 24. 本轮不做大量快筛

本轮直接训练：

```text
GRS-MambaSR ×4
500k
```

完成后统一测试：

```text
Set5
Set14
B100
Urban100
Manga109
```

只和固定 v1.0 baseline 做主比较。

---

# 25. 成功后再补必要消融

只有完整模型已经超过 baseline 后，再为论文补：

```text
w/o GRE
w/o GCR
w/o RAGA
Full model
```

最多 3~4 组。

这时消融实验用于证明机制，而不是用于寻找一个能涨点的模型。

---

# 26. 成功后再考虑 Text

只有：

\[
RGB+Depth>RGB
\]

稳定成立后，才考虑：

\[
RGB+Depth+Text
\]

并且 Text 不再直接对 RGB 做 global FiLM。

建议：

\[
Text\rightarrow Semantic\ Vector
\]

\[
Depth/RGB\rightarrow Spatial\ Reliability
\]

通过 geometry map 对 Text guidance 做 spatial grounding：

\[
G_T(x,y)=C_D(x,y)\cdot P(T)
\]

即：

> Depth 负责“在哪里”，Text 负责“是什么”。

如果 Text 最终不能继续提升，则直接放弃 Text；RGB+Depth 本身已经属于多模态 SR。

---

# 27. 风险评估

## 风险 1：Depth 来自 LR RGB，本身增量信息有限

如果：

\[
D=f(I_{LR})
\]

Depth 不是新的传感器观测，只提供 pretrained depth model 的结构归纳偏置。

因此合理预期应是：

```text
+0.05 ~ +0.20 dB
```

而不是期待 +0.5~1 dB。

## 风险 2：Depth edge 与 RGB texture edge 不一致

GRE 用于抑制错误注入。

## 风险 3：Routing bias 过强

通过：

```text
rho init = 0.05
small-init depth_route
reliability C
```

控制。

## 风险 4：模型增大但无增益

因此参数增量限制 <7%。

---

# 28. 最终建议

当前不继续 Text。

直接执行：

```text
v1.0 RGB MambaIRv2
          ↓
      GRS-MambaSR
          ↓
        500k
          ↓
Set5 / Set14 / B100 / Urban100 / Manga109
```

核心成功标准：

\[
\boxed{Avg\Delta PSNR>0}
\]

更推荐：

\[
\boxed{Avg\Delta PSNR\ge+0.05dB}
\]

并且：

\[
\boxed{Urban100/Manga109\ 有明确结构性提升}
\]

如果完整模型仍不能超过 baseline，不再无休止设计新的 Depth fusion 模块，而应重新评估 pseudo-depth 数据本身是否值得作为该任务的辅助模态。

---

# 29. 参考工作

1. MambaIRv2: Attentive State Space Restoration, CVPR 2025.
2. The Power of Context: How Multimodality Improves Image Super-Resolution, CVPR 2025.
3. Dual-Level Cross-Modality Neural Architecture Search for Guided Image Super-Resolution (DCNAS), TPAMI 2025.  
   https://github.com/zhwzhong/DCNAS
4. The Devil is in the Details: Boosting Guided Depth Super-Resolution via Rethinking Cross-Modal Alignment and Aggregation (D2A2).  
   https://github.com/JiangXinni/D2A2
5. IGAF: Incremental Guided Attention Fusion for Depth Super-Resolution, 2025.
6. DORNet: A Degradation Oriented and Regularized Network for Blind Depth Super-Resolution, CVPR 2025.
7. Completion as Enhancement: A Degradation-Aware Selective Image Guided Network for Depth Completion (SigNet), CVPR 2025.
8. SNUM-Net: Deep Semi-Smooth Newton-Driven Unfolding Network for Multi-Modal Image Super-Resolution, TIP 2025.  
   https://github.com/pandazcx/SNUM-Net
9. Joint Implicit Image Function for Guided Depth Super-Resolution (JIIF).  
   https://github.com/ashawkey/jiif
10. Guided Depth Super-Resolution by Deep Anisotropic Diffusion, CVPR 2023.  
    https://github.com/prs-eth/Diffusion-Super-Resolution
11. Learning Piecewise Planar Representation for RGB Guided Depth Super-Resolution (PGSR), TCI 2024.  
    https://github.com/XrKang/PGSR
