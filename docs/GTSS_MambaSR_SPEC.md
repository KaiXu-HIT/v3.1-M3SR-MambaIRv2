# GRS-MambaSR 三轮自我质疑后的优化方案
## 最终版本：GTSS-MambaSR
### Geometry-Transition Selective State-Space Mamba for RGB-Depth Image Super-Resolution

**基线仓库：** `KaiXu-HIT/v1.0-M3SR-MambaIRv2`  
**上一版 GRS 仓库：** `KaiXu-HIT/v3.0-M3SR-MambaIRv2`  
**任务：** RGB 主模态 + Depth 辅助模态，×4 classical image super-resolution  
**唯一性能锚点：** 超过 RGB-only MambaIRv2 baseline，而不是“证明多模态一定有效”。

---

# 1. 当前实验事实

固定 500k、相同训练配置下：

| Model | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---:|---:|---:|---:|---:|
| RGB baseline | 32.7645 | 29.0786 | 27.8639 | 27.3198 | 31.8458 |
| GRS v3.0 | 32.7342 | 29.0384 | 27.8687 | 27.3130 | 31.8735 |
| Δ | -0.0303 | -0.0402 | +0.0048 | -0.0068 | +0.0277 |

五库平均 PSNR 差约为：

\[
\Delta_{\mathrm{avg}}\approx -0.0090\text{ dB}
\]

因此 GRS v3.0 的正确结论不是“已经有效”，而是：

> **GRS 已经把此前明显的多模态负迁移压缩到接近 baseline 的水平，但尚未证明 Depth 带来了稳定的增量重建能力。**

这意味着下一版不能继续增加融合强度，而应进一步减少 Depth 对 RGB reconstruction manifold 的直接干预，同时让 Depth 只作用于它最有物理意义的位置。

---

# 2. 对 v3.0 代码的关键复核

v3.0 的 `grs_mambairv2_arch.py` 实际包含：

1. `DepthGeometryEncoder`：Depth + Sobel edge → 174-channel depth feature；
2. `GeometryReliabilityEstimator`：RGB edge / depth edge / edge difference / edge product → confidence；
3. `ReliabilityAwareGeometryAdapter (RAGA)`：
   \[
   F'=F+\alpha C A P(F_D)
   \]
4. `GeometryConditionedRouting (GCR)`：
   \[
   R'=R_{RGB}+\rho C R_D
   \]
5. RAGA 与 GCR 均作用于 ASSB 2/4/6。

同时，MambaIRv2 的 ASSM 实际执行：

\[
pred\_route=route(x)
\]

\[
cls\_policy=GumbelSoftmax(pred\_route, hard=True)
\]

然后根据 hard routing 的类别索引排序 token，再送入 `Selective_Scan`。

`Selective_Scan` 内部：

\[
x\_dbl \rightarrow (dts,B,C)
\]

其中 `dts` 经 `dt_projs_weight` 投影后，与 `dt_projs_bias` 一起交给 selective-scan kernel，并设置：

```python
delta_softplus=True
```

这两个代码事实直接决定了下一版应该如何修改。

---

# 3. 第一轮自我质疑：上一条“Depth 调 Δ/B”的方案是否真的比 GRS 更合理？

## 3.1 原始设想

上一轮提出：

\[
Depth \rightarrow Reliability \rightarrow (\Delta,B)
\]

删除 RAGA 和 GCR，让 Depth 连续调节 SSM dynamics。

这个方向比直接 feature fusion 更克制，但第一轮复核后发现仍有两个问题。

### 问题 A：同时调制 Δ 和 B 仍然过多

\(B_t\) 决定当前输入写入 state 的方式。Depth 是几何先验，不具备可靠的 RGB texture/color 信息。如果 Depth 同时修改：

- state retention（Δ）；
- input injection（B）；

仍然可能把 pseudo-depth 错误直接带入 RGB reconstruction dynamics。

### 问题 B：把二维 depth edge 直接广播到 selective scan 不严格

MambaIRv2 并不是按照普通 raster order 扫描。

它先：

1. hard semantic routing；
2. 按 routing index 排序；
3. 得到 `semantic_x`；
4. 在这个重新排序后的 1D 序列上 selective scan。

因此二维位置上的：

\[
|\nabla D(x,y)|
\]

并不等价于真正 scan path 上相邻两个 token 的几何变化。

## 第一轮修改

删除 \(B\) modulation。

只保留一个核心作用：

\[
\boxed{Depth\rightarrow \Delta}
\]

并要求 Depth guidance 必须与 **实际 semantic scan path 对齐**。

---

# 4. 第二轮自我质疑：怎样定义真正与 Mamba scan 对齐的 Depth 几何信号？

这是本轮最重要的修正。

## 4.1 原 GRS 的结构错位

原 GRS 的 confidence 是二维空间图：

\[
C(x,y)
\]

GCR 在 token flatten 后直接把它乘到 routing bias 上。

但 selective scan 的实际顺序是：

\[
x \xrightarrow{route} index
\xrightarrow{sort} semantic\_x
\]

也就是说，真正相邻发生状态递推的是：

\[
semantic\_x[t-1],\ semantic\_x[t]
\]

而不是原图中的二维相邻像素。

---

## 4.2 新定义：Scan-Path Geometry Transition

Depth 不再主要使用二维 Sobel edge 去控制 state transition。

先把 normalized depth：

\[
D\in[0,1]
\]

flatten 成 token：

\[
d\in\mathbb{R}^{B\times L}
\]

然后使用 **与 RGB token 完全相同的 `x_sort_indices`**：

\[
d^{s}=Gather(d,x\_sort\_indices)
\]

得到与 `semantic_x` 完全一致的 depth sequence。

定义真正的路径几何跳变：

\[
q_t=
|d_t^{s}-d_{t-1}^{s}|
\]

令：

\[
q_1=0
\]

这个 \(q_t\) 的含义非常明确：

> Mamba 即将从 scan-path 中前一个 token 向当前 token 传播 state 时，这两个 token 是否跨越了明显的深度层级。

这比二维 Sobel edge 更贴合当前 MambaIRv2 的真实计算图。

---

## 4.3 Reliability 也必须同步排序

保留 v3.0 已经实现的 GRE 思路，但把 confidence 同样排序：

\[
c^s=Gather(C,x\_sort\_indices)
\]

路径 transition reliability 定义为：

\[
r_t=\min(c_t^s,c_{t-1}^s)
\]

最终：

\[
g_t=r_t\cdot q_t
\]

其中：

\[
g_t\in[0,1]
\]

表示：

> **当前真实 Mamba scan transition 上，可信的几何不连续程度。**

## 第二轮修改

核心 guidance 从：

\[
2D\ Depth\ Edge
\]

升级为：

\[
\boxed{Semantic\ Scan\ Path\ Geometry\ Transition}
\]

这成为最终方案最重要的创新。

---

# 5. 第三轮自我质疑：怎样调 Δ 才不会再次破坏 baseline？

上一轮曾写：

\[
\Delta'=\Delta(1+\lambda g)
\]

但结合实际源码，这个公式不能直接无代价地实现。

原因是当前 MambaIRv2 把：

- raw `dts`；
- `dt_projs_bias`；

交给 fused selective-scan kernel，并设置：

```python
delta_softplus=True
```

也就是说实际有效 step size 是：

\[
\Delta_{\mathrm{eff}}
=
softplus(dts+b_\Delta)
\]

如果希望严格在 softplus 后做乘法：

\[
\Delta_{\mathrm{eff}}'
=
\Delta_{\mathrm{eff}}(1+\lambda g)
\]

就需要修改 fused kernel 接口或绕开当前实现，增加工程风险和效率损失。

---

## 5.1 最终改为 pre-softplus bounded transition bias

定义：

\[
u_t=\beta_i\cdot g_t
\]

然后：

\[
\boxed{
dts_t'=dts_t+u_t
}
\]

selective scan 内部仍然执行原来的：

\[
\Delta_t'
=
softplus(dts_t'+b_\Delta)
\]

因为 softplus 单调递增：

\[
g_t\uparrow
\Rightarrow
\Delta_t'\uparrow
\]

而 Mamba 中：

\[
A=-e^{A_{log}}<0
\]

所以：

\[
e^{\Delta_t'A}
\]

会减小，即跨越可信深度 discontinuity 时减少历史 state retention。

---

## 5.2 不允许 β 无界增长

使用：

\[
\beta_i=
\beta_{\max}\cdot sigmoid(\theta_i)
\]

建议：

\[
\beta_{\max}=0.5
\]

初始化时不设 0.05 的固定强扰动，而令：

\[
\beta_i\approx0.02
\]

对应：

\[
\theta_i=
logit(0.02/0.5)
\]

这样模型初始非常接近 RGB baseline，但 Depth 路径仍然有非零梯度。

---

## 5.3 Channel-wise 还是 scalar？

最初考虑为每个 hidden channel 学一个 \(\beta_c\)，但这会降低几何机制的可解释性，并允许网络把 Depth gate 变成复杂的 channel modulation。

最终采用：

\[
\boxed{\text{每个被调制 ASSM 一个 scalar }\beta}
\]

然后广播到 hidden channels。

这是第三轮收缩后的选择。

---

# 6. 三轮迭代后的最终模型：GTSS-MambaSR

## 名称

**GTSS-MambaSR**

**Geometry-Transition Selective State-Space Mamba for RGB-Depth Image Super-Resolution**

核心思想：

> Depth 不再生成 RGB residual，不改变 hard semantic routing，也不直接修改 B/C state projections；Depth 只在 MambaIRv2 已经确定的 semantic scan path 上计算几何 transition，并连续调节该 transition 的 state-retention step。

最终信息流：

```text
RGB ────────────────────────────────────────────────────────────┐
 │                                                              │
 ├─ RGB edge ─┐                                                 │
 │            │                                                 │
 │            ├─ Geometry Reliability C                         │
 │            │                                                 │
Depth ─ Sobel ┘                                                 │
 │                                                              │
 └─ normalized depth tokens d                                   │
                                                                │
RGB feature → route → hard policy → sort index ────────────┐    │
                                                           │    │
depth d ───────── same sort index ─→ d^s ─→ |d_t-d_t-1|    │    │
confidence C ─── same sort index ─→ c^s ─→ pair reliability│    │
                                                           ▼    │
                                  Path Geometry Gate g_t         │
                                           │                    │
                                           ▼                    │
                                  dts' = dts + β g_t             │
                                           │                    │
                                           ▼                    │
                                     Selective Scan             │
                                           │                    │
                                           └──────────────→ RGB SR
```

---

# 7. 最终保留和删除的组件

| v3.0 GRS component | GTSS 决策 | 原因 |
|---|---|---|
| P2/P98 depth normalization | 保留 | 对 monocular/pseudo-depth 尺度更稳健 |
| Sobel RGB/depth edge | 保留 | 仅用于 reliability |
| GRE | 保留但轻量化 | 过滤明显不一致 guidance |
| 174-ch DGE | 删除 | 不再需要生成 RGB-like depth feature |
| RAGA | 删除 | 直接污染 RGB feature 的风险 |
| GCR | 删除 | hard routing 对小 bias 不连续 |
| Depth→B modulation | 删除 | Depth 不应控制 RGB input content 写入 |
| Depth→Δ | 保留并重构 | 与 state retention 直接对应 |
| 2D edge→Δ | 删除 | 与 semantic scan order 不一致 |
| scan-path depth transition | 新增 | 与真实 SSM transition 对齐 |

---

# 8. Depth 分支进一步简化

最终甚至不再需要原来的 `DepthGeometryEncoder(dim=174)`。

Depth 分支只提供两种东西：

1. normalized scalar depth \(D\)；
2. depth Sobel edge \(E_D\) 用于 reliability。

因此新增参数主要来自 GRE 和极少量 \(\beta\)。

---

# 9. GRE 最终形式

输入仍为：

\[
Z=[
E_R,\,
E_D,\,
|E_R-E_D|,\,
E_R E_D
]
\]

网络缩小：

```text
4 ch
 ↓
Conv 3×3: 4 → 16
 ↓
GELU
 ↓
Conv 3×3: 16 → 8
 ↓
GELU
 ↓
Conv 1×1: 8 → 1
 ↓
Sigmoid
 ↓
C
```

这样 reliability estimator 本身不具备足够容量去偷偷承担第二个 feature encoder 的角色。

---

# 10. Path Geometry Gate 的精确定义

设：

\[
D_f=Flatten(D)
\]

\[
C_f=Flatten(C)
\]

使用 ASSM 当前已经产生的：

\[
x\_sort\_indices
\]

得到：

\[
D_s=Gather(D_f,x\_sort\_indices)
\]

\[
C_s=Gather(C_f,x\_sort\_indices)
\]

然后：

\[
Q_t=|D_s(t)-D_s(t-1)|
\]

为避免少量异常值：

\[
\hat Q_t=clamp(Q_t,0,q_{max})/q_{max}
\]

建议固定：

\[
q_{max}=0.25
\]

原因：Depth 已经 P2/P98 normalize 到 [0,1]，0.25 表示较明显的相对深度层变化；超过该值不需要继续线性增加 state suppression。

可靠性：

\[
R_t=\min(C_s(t),C_s(t-1))
\]

最终：

\[
\boxed{
G_t=R_t\hat Q_t
}
\]

并令：

\[
G_1=0
\]

---

# 11. 为什么不用 learned threshold 网络

可以设计：

\[
G=MLP(D_t,D_{t-1},C_t,C_{t-1})
\]

但最终不采用。

原因：

1. 参数更多；
2. 可解释性下降；
3. 更容易学习 dataset-specific shortcut；
4. 当前目标是保护强 baseline；
5. 明确的 depth difference 已经对应所需物理量。

第一版 GTSS 应保持机制尽可能确定。

---

# 12. 如何接入 Selective_Scan

当前：

```python
dts = torch.einsum(...)
...
out_y = self.selective_scan(
    xs, dts,
    As, Bs, Cs, Ds,
    delta_bias=dt_projs_bias,
    delta_softplus=True,
)
```

GTSS 修改为概念上的：

```python
if geometry_transition is not None:
    # geometry_transition: [B, 1, L]
    beta = beta_max * torch.sigmoid(beta_logit)
    dts = dts + beta * geometry_transition.unsqueeze(1)

out_y = self.selective_scan(
    xs, dts,
    As, Bs, Cs, Ds,
    delta_bias=dt_projs_bias,
    delta_softplus=True,
)
```

实际 shape 应严格匹配当前：

\[
dts:[B,K,d_{inner},L]
\]

所以 gate 应 reshape/broadcast 为：

\[
[B,1,1,L]
\]

再广播到 \(d_{inner}\)。

---

# 13. Geometry gate 必须在排序之后生成

不要在 `GRSMambaIRv2.forward_features()` 中提前生成完整 transition map。

因为：

\[
x\_sort\_indices
\]

是在每个 ASSM 内由该 ASSM 自己的 routing dictionary 产生的。

不同 ASSM：

\[
x\_sort\_indices^{(j)}
\]

可以不同。

所以正确实现位置必须在：

```python
ASSM.forward()
```

内部：

```text
pred_route
→ cls_policy
→ x_sort_indices
→ reorder RGB feature
→ reorder depth
→ reorder confidence
→ compute transition
→ selective scan
```

这一点是本次代码复核后非常重要的修正。

---

# 14. 每个 ASSM 都调制，还是只调一部分？

三轮质疑后，不建议继续沿 v3.0 的“ASSB 2/4/6 中所有 ASSM 都强制加入复杂 guidance”。

最终建议：

## 在 ASSB 2、4、6 中启用 GTSS，但每个 ASSM 都使用独立 β

原因：

- transition index 本来就是每个 ASSM 独立产生；
- gate 本身几乎零参数；
- 如果只在最后一个 ASSM 使用，会把 Depth 的作用压缩得过弱；
- 但通过 β 的小初始化和上界限制，能够避免 v3.0 那种 feature+route 双重扰动。

因此共：

\[
3\times6=18
\]

个 scalar \(\beta\)。

总新增参数几乎可以忽略。

---

# 15. 为什么不直接 6 个 ASSB 全部使用

ASSB 1/3/5 保持纯 RGB，形成：

```text
RGB stage → geometry-conditioned stage
RGB stage → geometry-conditioned stage
RGB stage → geometry-conditioned stage
```

即交替式：

\[
R-G-R-G-R-G
\]

这给网络留下纯 RGB representation refresh 的空间，也保持与 v3.0 相同的 stage-level intervention density，减少比较变量。

---

# 16. 初始化

设：

\[
\beta_{\max}=0.5
\]

希望初始：

\[
\beta_0=0.02
\]

因此：

\[
\theta_0=
\log\frac{0.02}{0.5-0.02}
\approx -3.178
\]

代码：

```python
beta_max = 0.5
beta_logit = nn.Parameter(torch.tensor(-3.178))
beta = beta_max * torch.sigmoid(beta_logit)
```

优点：

- 初始不是完全关闭；
- 有梯度；
- 最大扰动严格有界；
- 不允许训练后 β 爆炸。

---

# 17. Loss：继续保持 baseline L1

\[
\mathcal L=\|I_{SR}-I_{HR}\|_1
\]

不增加：

- edge loss；
- depth loss；
- perceptual；
- GAN；
- frequency loss。

因为当前实验要验证的是：

\[
\boxed{Depth-conditioned state transition 本身能否提高 fidelity}
\]

---

# 18. 训练策略

继续使用与 baseline / v3.0 相同的完整训练配置：

```text
Scale = ×4
total_iter = 500000
GT patch = 192
LR patch = 48
Adam
lr = 2e-4
betas = [0.9, 0.99]
L1 loss
same scheduler
same DIV2K training data
same augmentation
```

**直接完整训练，不做 Add/Concat/FiLM/多位置快筛。**

---

# 19. 测试策略

保持：

```text
Set5
Set14
B100
Urban100
Manga109
```

主比较：

```text
v1.0 RGB baseline
v3.0 GRS
GTSS-MambaSR
```

由于 MambaIRv2 原实现 eval 中仍使用 hard Gumbel softmax，最终 GTSS 与 baseline 建议各重复 3 次 inference，报告 mean/std；这不属于额外模型训练。

---

# 20. 成功标准

第一目标不是超过 v3.0，而是超过 v1.0。

最低：

\[
Avg\Delta PSNR>0
\]

值得继续：

\[
Avg\Delta PSNR\ge +0.05\text{ dB}
\]

并希望：

\[
Urban100\ge +0.08\text{ dB}
\]

或：

\[
Manga109\ge +0.10\text{ dB}
\]

同时 B100 不明显下降。

---

# 21. 为什么预期 Urban100 / Manga109 更敏感

GTSS 不直接补 RGB texture，而主要抑制不合理的跨几何层状态传播。

因此更可能受益于：

- 建筑线条；
- 重复窗格；
- 漫画轮廓；
- 前景/背景清晰分界；
- 大量规则结构。

所以如果最终表现为：

```text
Set5       ~ 0
Set14      small +
B100       small +
Urban100   clear +
Manga109   clear +
```

这反而与方法机制高度一致。

---

# 22. 相比 v3.0，GTSS 的关键优势

## 22.1 更 baseline-preserving

v3.0：

\[
Depth\rightarrow Feature + Routing
\]

GTSS：

\[
Depth\rightarrow \Delta\ only
\]

---

## 22.2 避免 hard-routing discontinuity

v3.0：

\[
R+\epsilon\rightarrow argmax
\]

可能发生离散 route flip。

GTSS：

\[
dts+\beta G
\rightarrow softplus
\]

是连续变化。

---

## 22.3 与实际 scan path 对齐

不是：

\[
2D\ edge
\]

直接指导 1D state scan。

而是：

\[
\boxed{
Depth\ reordered\ by\ the\ exact\ semantic\ scan\ index
}
\]

再计算：

\[
|D_t-D_{t-1}|
\]

---

## 22.4 参数更少

不再需要：

- 174-channel DGE；
- 3 个 RAGA；
- 3 个 174→128 route projection。

只保留：

- tiny GRE；
- 18 个 β scalar。

---

# 23. 最终论文核心创新点

如果实验成功，建议不要包装成三个拼接模块，而围绕一个核心命题：

> **Semantic routing organizes tokens according to appearance/semantic affinity, but such a 1D sequence can still connect tokens from different geometric layers. GTSS introduces depth-derived transition priors directly along the actual semantic scan path to regulate state retention without altering RGB features or semantic routing.**

可凝练为两个贡献。

### Contribution 1：Scan-Path Geometry Transition Prior

首次不是在原二维坐标上简单融合 depth，而是根据 MambaIRv2 的动态 semantic routing，把 Depth 同步重排到实际 scan path：

\[
D\rightarrow D^s
\]

并显式计算：

\[
|D_t^s-D_{t-1}^s|
\]

作为 state transition geometry prior。

### Contribution 2：Reliability-Guided Continuous State Regulation

利用 RGB-depth edge consistency 得到 reliability，再对 transition prior 进行过滤，并通过 bounded pre-softplus Δ bias 连续调节 SSM memory decay：

\[
dts'=dts+\beta G
\]

避免直接 feature contamination 和 hard-routing flip。

---

# 24. 三轮迭代总结

## Iteration 1

原方案：

\[
Depth\rightarrow(\Delta,B)
\]

质疑：

- B 会让 Depth 影响 RGB content injection；
- 仍可能过强。

修改：

\[
Depth\rightarrow\Delta\ only
\]

---

## Iteration 2

原方案：

\[
2D\ depth\ edge\rightarrow\Delta
\]

质疑：

- MambaIRv2 实际在 semantic-sorted 1D sequence 上 scan；
- 2D edge 与真实 state transition 不对齐。

修改：

\[
Depth
\rightarrow same\ semantic\ sorting
\rightarrow |D_t-D_{t-1}|
\rightarrow\Delta
\]

---

## Iteration 3

原方案：

\[
\Delta'=\Delta(1+\lambda G)
\]

质疑：

- 当前 fused selective scan 在 kernel 内做 `delta_softplus=True`；
- post-softplus 乘法会增加实现复杂度；
- 无界/通道级调制可能重新造成过拟合。

修改：

\[
\boxed{
dts'=dts+\beta G
}
\]

并采用：

\[
0<\beta<0.5
\]

的 bounded scalar modulation。

---

# 25. 最终决策

三轮自我质疑之后，我认为比原 GRS v3.0 更值得投入下一次 500k 完整训练的不是“更复杂 GRS”，而是：

\[
\boxed{
\text{GTSS-MambaSR}
}
\]

最终核心链路只有：

\[
\boxed{
Depth
\rightarrow
Reliability
\rightarrow
Semantic-Path Depth Transition
\rightarrow
Bounded \Delta Modulation
\rightarrow
Selective Scan
}
\]

它删除了 v3.0 中最可能引起负迁移的两条路径：

\[
\boxed{RAGA}
\]

和：

\[
\boxed{GCR}
\]

同时没有退化成普通的“Depth gate”。

它把 Depth 的几何信息直接映射到 Mamba 最符合几何含义的变量——**实际 scan transition 的 state retention**。

---

# 26. 明确停止规则

如果 GTSS 完整 500k 后仍然：

\[
Avg\Delta PSNR\le0
\]

且 Urban100 / Manga109 没有明确正增益，则不建议继续设计第四套 RGB+Depth Mamba fusion。

届时更合理的科学判断是：

> 当前 pseudo-depth 在该 DIV2K bicubic ×4 fidelity-SR 设置下，对强 MambaIRv2 baseline 的增量信息不足。

后续应转向：

- 更高质量/独立来源的 depth；
- 其他空间先验；
- 或重新定义 multimodal SR 的评价目标；

而不是继续堆叠 cross-modal 模块。

---

# 27. 推荐源码修改边界

以 `v3.0-M3SR-MambaIRv2` 为基础：

### 删除/停用

```text
DepthGeometryEncoder
ReliabilityAwareGeometryAdapter
GeometryConditionedRouting
geometry_route_bias
depth_feat
```

### 保留并简化

```text
SobelGradient
robust_normalize
GeometryReliabilityEstimator
```

### 新增

```text
GeometryTransitionController
```

但它不需要独立 feature network；核心工作在 `ASSM.forward()` 内完成：

```text
depth/confidence
→ same x_sort_indices
→ pairwise transition
→ beta
→ Selective_Scan.forward_core()
```

### 修改函数接口

概念上：

```python
ASSM.forward(
    x,
    x_size,
    token,
    depth=None,
    confidence=None,
    enable_gtss=False
)
```

以及：

```python
Selective_Scan.forward(
    x,
    prompt,
    geometry_transition=None,
    beta=None
)
```

这是相对 v3.0 较小、可控、可直接实现的一次结构修改。
