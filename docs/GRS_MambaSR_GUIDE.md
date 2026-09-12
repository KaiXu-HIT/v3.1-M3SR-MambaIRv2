# GRS-MambaSR v3.0

## Scope and provenance

Built from the pure RGB baseline commit `d84240a`. The pre-task checkout is
preserved locally on `backup/pre-grs-20260912` (`4bc8435`). No early-concat,
Depth Identity, text, FiLM, second Mamba stream, or extra loss is introduced.
The attached model specification is the implementation contract.

## Implementation map

| Specification | Implementation |
|---|---|
| DGE | Fixed Sobel depth gradient; [D,E_D], 2->48 convolution, two DW/PW blocks, 48->174 convolution |
| GRE | Independently P2/P98-normalized LR RGB luminance/depth edges; [E_R,E_D,abs(E_R-E_D),E_R*E_D]; 4->32->16->1 sigmoid |
| RAGA | At ASSB 2/4/6 only, before the stage; 1x1/GELU/DW3x3/1x1 residual; RGB GAP channel gate; alpha=0.05 |
| GCR | At ASSB 2/4/6 only; rho*C_t*Linear(F_D) added to original RGB route before unchanged hard Gumbel Softmax |
| Initialization | RAGA final conv and GCR linear normal std=1e-3, bias=0; rho=0.05; DGE Kaiming |
| Backbone | Original six ASSBs, RGB dictionaries, window attention, prompt/sort/scan/fold, body residual and pixelshuffle head retained |
| Data | Original server roots and filename templates; full-image depth P2/P98 before shared crop/flip/rotation |
| Training | DIV2K, RGB x4 bicubic, GT192/LR48, batch=2 on one GPU, seed10, L1, Adam 2e-4, 500k |
| Scheduler | MultiStepLR [250000,400000,450000,475000], gamma=0.5 |
| Evaluation | Set5/Set14/B100/Urban100/Manga109; baseline chop/overlap; border4, Y PSNR/SSIM |
| Diagnostics | alpha_2/4/6, rho_2/4/6, confidence_mean/std, depth_residual_abs_mean |

The specification leaves the channel-gate bottleneck and stage sharing details
open. The implementation uses C//16 with GELU for the channel gate, and one
linear depth projection plus rho per selected ASSB, reused by all six ASSMs
inside that stage. This implements the stage-indexed rho_i and W_D equations
without introducing another routing network. All RGB route parameters remain
per ASSM. Geometry edges are estimated before window padding; feature and
confidence maps then use the same symmetric padding as RGB.

The baseline architecture changes only to accept an optional geometry route
bias. Omitting it executes the original RGB path. Both guidance scales set to
zero must reproduce the original d84240a model under the same Gumbel seed.

Depth files keep their existing image format (including 16-bit images); RGB
and depth must have identical LR dimensions. There is no implicit resizing or
HR-derived depth generation. Missing pairs and invalid depth values fail early.
No dataset payload, model checkpoint, or local virtual environment is committed.

## Server commands

Use the existing Linux CUDA/MambaIR environment. The Windows local project is
`D:\Code\Python\M3SR-MambaIRv2`; the following checkout command creates the
corresponding server project alongside the existing v1.0 directory.

```bash
cd /home/BRAIN/xukai/code
git clone https://github.com/KaiXu-HIT/v3.0-M3SR-MambaIRv2.git
cd v3.0-M3SR-MambaIRv2
conda activate mambair
```

If already cloned, enter that directory and run `git pull --ff-only origin main`.
Use the name of your existing baseline environment if it is not `mambair`.
Do not replace the server's working CUDA/Mamba environment with the local CPU
verification environment.

Preflight (actual CUDA scan forward/backward; filename checks for all pairs and
first/middle/last decoding in DIV2K train/valid and all five benchmarks):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/grs/check_grs.py --check-data
```

Train from scratch (formal one-GPU baseline-matched recipe):

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_GRS_MambaSR_x4.yml
```

Resume an interrupted run, preserving optimizer/scheduler state:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/mambairv2/train_GRS_MambaSR_x4.yml --auto_resume
```

Evaluate the final 500k checkpoint on all five datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_GRS_MambaSR_x4.yml
```

Optional explicit checkpoint selection (do not use benchmark scores to select
among multiple checkpoints for the formal comparison):

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/mambairv2/test_GRS_MambaSR_x4.yml --force_yml path:pretrain_network_g=experiments/v3.0_GRS_MambaSR_x4/models/net_g_500000.pth
```

Paired three-run comparison with the existing fixed RGB checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/grs/evaluate_repeated.py --seeds 10 11 12
```

Default baseline checkpoint (unchanged from the prior evaluation configuration):
`/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth`.
Default GRS checkpoint:
`experiments/v3.0_GRS_MambaSR_x4/models/net_g_500000.pth`.
Override paths with `--baseline-checkpoint PATH --grs-checkpoint PATH` when
necessary. This does not warm-start GRS training.

The paired script runs six separate processes. Each pair uses the same seed;
the three pairs use 10/11/12 so stochastic inference variability is observable.
It resets RNG after model construction, sorts both datasets by LR filename,
and uses the same deterministic cuDNN setting for both models. The original
Gumbel mechanism remains active. Therefore use the newly measured baseline
mean when computing deltas; old single-run published values are only context.

Output: `results/grs_comparison/summary.md` and `summary.json`, including the
six per-run results, PSNR/SSIM mean and sample standard deviation (ddof=1),
five-dataset mean PSNR delta and nondecreasing-dataset count. Minimum effective
criterion: mean delta >=0.03 dB and at least 4/5 datasets not decreasing.

## Parameter count

- Original RGB backbone: 23,050,713 parameters.
- GRS-MambaSR: 23,404,574 parameters.
- Added: 353,861 parameters (1.5351%), below 1.5M and 7%.

## Verification boundary

The local CPU checks use an explicit differentiable selective-scan reference
recurrence and isolated architecture loading to avoid unavailable CUDA imports.
They exercise geometry gradients, original-baseline identity, parameter limits,
16-bit depth alignment/augmentation, and partition stitching. They do not prove
CUDA binary compatibility or dataset provenance. Run the server preflight above
in the existing training environment. This delivery does not run 500k training
or assert any benchmark improvement.

```powershell
.\.venv\Scripts\python.exe scripts/grs/check_grs.py --cpu-reference
```
