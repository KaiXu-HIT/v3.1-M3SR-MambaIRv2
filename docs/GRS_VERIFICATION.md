# Local verification report

Date: 2026-09-12. Source baseline: d84240a. Windows CPU verification only.

Command: `.venv/Scripts/python.exe scripts/grs/check_grs.py --cpu-reference`.

Passed:

- Exact baseline optimizer, scheduler, loss, iteration and existing RGB dataset options match.
- Full production model: 174 channels, 6x6 attentive layers, RGB16x16 -> SR64x64 forward, finite output.
- Six-stage reduced model: RGB9x11 -> SR36x44, L1 backward and Adam step; all DGE/GRE/RAGA/GCR parameter gradients present and finite; every branch has nonzero gradient.
- Alpha/rho initialized to 0.05; RAGA/GCR final projections have std approximately 1e-3.
- Alpha=rho=0: bit-identical output to original d84240a source under matched Gumbel RNG on CPU.
- Strict state-dictionary load and same-seed output reproduction; tiny RGB1x2 -> SR4x8.
- Missing/misaligned Depth rejected; constant-edge reliability finite and in [0,1].
- Synthetic 16-bit PNG loading, full-image P2/P98 normalization, shared crop/flip/rotation under eight seeds.
- Original baseline and GRS partition stitching match exactly on RGB203x407; depth slices align; normal and EMA branches checked.
- Three-seed evaluation orchestration/statistics checked with synthetic metric fixtures: six correctly paired jobs, sample std, mean delta and threshold evaluation. These are software checks, not benchmark results.
- All six changed/new Python sources compiled; source/config git diff whitespace checks passed. The archived original specification retains its Markdown two-space hard line breaks.

Parameters:

| Model | Parameters |
|---|---:|
| RGB baseline | 23,050,713 |
| GRS-MambaSR | 23,404,574 |
| Added | 353,861 (1.5351%) |

The isolated CPU loader uses PyTorch tensor operators and an explicit selective
scan recurrence, not the production CUDA extension. Baseline source arithmetic
and production GRS source are executed directly; only optional imports are
isolated. The isolated environment is excluded from Git.

Not executed locally: Linux CUDA extension integration, server dataset access,
full 500k training, PSNR/SSIM benchmark inference, or confirmation of Depth
provenance. No accuracy improvement is claimed. Run the CUDA/data preflight and
training/evaluation commands in GRS_MambaSR_GUIDE.md on the original server.
