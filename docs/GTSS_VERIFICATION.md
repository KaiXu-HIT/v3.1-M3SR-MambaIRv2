# GTSS v3.1 verification — 2026-09-15

Local environment: Windows, isolated CPU PyTorch 2.14.0+cpu.
Source base: v3.0 commit 697f97a. Original RGB reference: d84240a.

## Passed

- Full config: 174 channels, 36 attentive layers; finite 16x16 LR -> 64x64 output.
- Exactly 18 independent scalar beta parameters on ASSB 2/4/6; beta initially 0.02, max factor 0.5, q_max=0.25.
- Tiny GRE widths 4/16/8/1; no DGE/RAGA/GCR in the instantiated GTSS network.
- Parameters: RGB 23,050,713; GTSS 23,052,492; difference 1,779 (0.0077178%).
- Hand-computed non-raster depth permutation: correct endpoint minimum, normalized/clipped differences, first-token zero, constant-depth and zero-confidence gates.
- Captured selective-scan arguments: only raw dts changes by beta*G; two-image batch and channel broadcasting correct; A/B/C/input/D/delta_bias unchanged; delta_softplus=True; effective delta increases and exp(delta*A) does not increase.
- Single-ASSM same-seed depth perturbation leaves routing permutation, prompt and scan input identical, while changing scan output.
- Production optimize_parameters method on a 36-layer reduced-width network: L1 backward, all 18 beta gradients and all GRE parameter gradients nonzero/finite on first step; Adam and diagnostics work.
- Constant-depth GTSS matches original d84240a output bit-for-bit under identical RNG. Active RGB-only model also matches d84240a.
- Strict serialized state-dictionary round-trip, same-seed reproduction and tiny 1x2 input.
- Unchanged model test-method AST plus odd-size partition/overlap stitching and matching depth slices (normal/EMA).
- Unchanged 16-bit Depth reading/P2-P98 normalization, synchronized crop/flips/rotation, invalid alignment/missing-file failures.
- Train/test dataset dictionaries unchanged from v3.0; training loss/optimizer/scheduler/500k configuration unchanged.
- Frozen v3.0 backbone source preserved except removal of automatic architecture registration; original GRS regression passes.
- Three-model mean/sample-std and RGB-referenced criteria tested with synthetic fixtures, not reported as benchmark measurements.
- CLI orchestration: default six jobs and optional nine jobs are seed-paired; reserved GRS path fails before any worker starts; JSON/Markdown summaries validated.
- All 10 new/changed Python files compile; source/config whitespace checks pass. Original specification Markdown hard line breaks are preserved.

Commands:

```powershell
.\.venv\Scripts\python.exe scripts/gtss/check_gtss.py --cpu-reference
.\.venv\Scripts\python.exe scripts/grs/check_grs.py --cpu-reference
```

CPU isolation bypasses optional CUDA imports but executes actual architecture
source and a differentiable reference recurrence. It does not validate the fused
CUDA binary, production BasicSR imports on the server, or server image payloads.
No 500k training or real PSNR/SSIM evaluation was run for this delivery. Run
`python scripts/gtss/check_gtss.py --check-data` in the original CUDA environment
before starting the formal experiment. No performance gain is asserted.
