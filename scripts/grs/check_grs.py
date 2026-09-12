"""GRS contract checks. CUDA mode exercises the installed BasicSR/Mamba stack.

--cpu-reference isolates architecture source from optional CUDA imports and uses
an explicit differentiable selective-scan recurrence. This is not a CUDA-kernel
or server-data validation. It never changes the production scan implementation.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    # Exact real-valued selective scan recurrence for the baseline's K=1 path.
    dt = delta.float()
    if delta_bias is not None:
        dt = dt + delta_bias[None, :, None]
    if delta_softplus:
        dt = F.softplus(dt)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        state = (dt[:, :, t, None] * A).exp() * state + (
            dt[:, :, t, None] * B[:, 0, :, t][:, None, :] * u[:, :, t, None])
        outputs.append((state * C[:, 0, :, t][:, None, :]).sum(-1))
    y = torch.stack(outputs, -1)
    if D is not None:
        y = y + u * D[None, :, None]
    if z is not None:
        y = y * F.silu(z)
    return (y, state) if return_last_state else y


def load_isolated(source, extra=None):
    import math
    class Registry:
        def register(self):
            return lambda cls: cls
    env = dict(torch=torch, nn=nn, F=F, np=np, math=math, rearrange=rearrange,
               repeat=repeat, selective_scan_fn=reference_scan,
               selective_scan_ref=reference_scan, ARCH_REGISTRY=Registry(),
               to_2tuple=lambda x: x if isinstance(x, (tuple, list)) else (x, x),
               trunc_normal_=nn.init.trunc_normal_, __name__='grs_isolated_check')
    if extra:
        env.update(extra)
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, '<isolated source>', 'exec'), env)
    return types.SimpleNamespace(**env)


def check_dataset_locally():
    import cv2
    # Exercise real dataset and transform source on synthetic 16-bit PNG pairs.
    source = (ROOT/'basicsr/data/rgb_depth_paired_image_dataset.py').read_text()
    transforms = load_isolated((ROOT/'basicsr/data/transforms.py').read_text(),
                               dict(cv2=cv2, random=random))
    class Registry:
        def register(self):
            return lambda cls: cls
    class FileClient:
        def __init__(self, *a, **kw): pass
        def get(self, path, key): return Path(path).read_bytes()
    def decode(content, flag='color', float32=False):
        x = cv2.imdecode(np.frombuffer(content, np.uint8),
                         cv2.IMREAD_UNCHANGED if flag == 'unchanged' else cv2.IMREAD_COLOR)
        return x.astype(np.float32)/255 if float32 else x
    def tensor(images, **kwargs):
        return [torch.from_numpy(np.ascontiguousarray(
            x[..., ::-1] if x.shape[-1] == 3 else x)).permute(2, 0, 1).float() for x in images]
    mod = load_isolated(source, dict(osp=__import__('os').path, data=torch.utils.data,
        DATASET_REGISTRY=Registry(), FileClient=FileClient, imfrombytes=decode,
        img2tensor=tensor, scandir=lambda p: [x.name for x in Path(p).iterdir() if x.is_file()],
        paired_random_crop=transforms.paired_random_crop, augment=transforms.augment))
    with tempfile.TemporaryDirectory() as directory:
        d = Path(directory)
        for name in ('gt','lq','depth'): (d/name).mkdir()
        scalar = np.arange(16*16, dtype=np.uint8).reshape(16,16)
        rgb = np.repeat(scalar[...,None],3,2)
        cv2.imwrite(str(d/'lq/a.png'),rgb)
        cv2.imwrite(str(d/'gt/a.png'),rgb.repeat(4,0).repeat(4,1))
        cv2.imwrite(str(d/'depth/a.png'),scalar.astype(np.uint16)*257)
        opt = dict(dataroot_gt=str(d/'gt'),dataroot_lq=str(d/'lq'),dataroot_depth=str(d/'depth'),
                   io_backend={'type':'disk'},phase='train',scale=4,gt_size=32,use_hflip=True,use_rot=True)
        ds = mod.RGBDepthPairedImageDataset(opt)
        lo,hi=np.quantile(scalar.astype(np.float32)*257,[.02,.98])
        for seed in range(8):
            random.seed(seed)
            v=ds[0]
            torch.testing.assert_close(v['gt'][:,::4,::4],v['lq'])
            expected=((v['lq'][0:1]*255*257-lo)/(hi-lo+1e-6)).clamp(0,1)
            torch.testing.assert_close(v['depth'],expected)
        cv2.imwrite(str(d/'depth/a.png'),np.zeros((8,8),np.uint16))
        try: ds[0]
        except ValueError: pass
        else: raise AssertionError('Misaligned depth was accepted')
        (d/'depth/a.png').unlink()
        try: mod.RGBDepthPairedImageDataset(opt)
        except FileNotFoundError: pass
        else: raise AssertionError('Missing depth was accepted')
    print('PASS: 16-bit depth normalization, shared crop/augmentation, missing/misaligned rejection')


def check_partition():
    # The exact production test methods with simple nearest-neighbor nets isolate
    # partition/overlap stitching from architecture costs on large odd images.
    def test_method(path):
        tree=ast.parse((ROOT/path).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='test')
        env=dict(torch=torch,F=F)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        return env['test']
    baseline=test_method('basicsr/models/mambairv2_model.py')
    grs=test_method('basicsr/models/grs_mambairv2_model.py')
    class Toy(nn.Module):
        def forward(self,x,depth=None):
            if depth is not None:
                torch.testing.assert_close(x[:,0:1],depth)
            return F.interpolate(x,scale_factor=4,mode='nearest')
    for ema in (False,True):
        rgb=torch.rand(1,3,203,407)
        a=types.SimpleNamespace(lq=rgb,depth=rgb[:,0:1],opt={'scale':4},net_g=Toy())
        if ema: a.net_g_ema=Toy()
        grs(a); got=a.output.clone()
        baseline(a)
        torch.testing.assert_close(got,a.output,rtol=0,atol=0)
        torch.testing.assert_close(got,F.interpolate(rgb,scale_factor=4,mode='nearest'),rtol=0,atol=0)
    print('PASS: baseline-identical odd-size RGB/depth partition stitching, normal and EMA')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-reference',action='store_true')
    p.add_argument('--check-data',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)
    if a.cpu_reference:
        base=load_isolated((ROOT/'basicsr/archs/mambairv2_arch.py').read_text())
        grs=load_isolated((ROOT/'basicsr/archs/grs_mambairv2_arch.py').read_text(),dict(MambaIRv2=base.MambaIRv2))
        device='cpu'
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; local checks require --cpu-reference.')
        from basicsr.archs import mambairv2_arch as base, grs_mambairv2_arch as grs
        device='cuda'
    cfg=yaml.safe_load((ROOT/'options/train/mambairv2/train_GRS_MambaSR_x4.yml').read_text())
    baseline_cfg=yaml.safe_load((ROOT/'options/train/mambairv2/train_S0_RGB_MambaIRv2_x4.yml').read_text())
    assert cfg['train']==baseline_cfg['train']
    for phase in ('train','val'):
        for field,value in baseline_cfg['datasets'][phase].items():
            if field not in ('name','type'):
                assert cfg['datasets'][phase][field]==value,(phase,field)
    netopt=copy.deepcopy(cfg['network_g']);netopt.pop('type')
    full=grs.GRSMambaIRv2(**netopt)
    pure=base.MambaIRv2(**netopt)
    count=lambda m: sum(x.numel() for x in m.parameters())
    n0,n1=count(pure),count(full)
    assert n1-n0<1500000 and (n1-n0)/n0<.07
    assert set(full.adapters)=={'2','4','6'}
    for m in full.adapters.values():
        assert abs(m.alpha.item()-.05)<1e-7
        assert .0009<m.depth_proj[-1].weight.std().item()<.0011
    for m in full.geometry_routes.values():
        assert abs(m.rho.item()-.05)<1e-7
        assert .0009<m.depth_route.weight.std().item()<.0011
    print(json.dumps(dict(baseline_params=n0,grs_params=n1,added=n1-n0,added_percent=100*(n1-n0)/n0)))
    # Also exercise all 36 attentive layers with the production channel widths.
    full=full.to(device)
    side=16 if a.cpu_reference else 48
    with torch.no_grad():
        full_output=full(torch.rand(1,3,side,side,device=device),
                         torch.rand(1,1,side,side,device=device))
    assert full_output.shape==(1,3,side*4,side*4) and torch.isfinite(full_output).all()
    print('PASS: full 174-channel / 36-layer production model forward')
    del full,pure,full_output
    small=dict(img_size=8,embed_dim=12,d_state=4,depths=[1]*6,num_heads=[3]*6,
               window_size=4,inner_rank=4,num_tokens=8,convffn_kernel_size=5,
               mlp_ratio=2.,upscale=4,upsampler='pixelshuffle')
    rgb=torch.rand(1,3,9,11,device=device);depth=torch.rand(1,1,9,11,device=device)
    model=grs.GRSMambaIRv2(**small).to(device)
    target=torch.rand(1,3,36,44,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.9,.99))
    torch.manual_seed(10)
    out=model(rgb,depth)
    assert out.shape==target.shape and torch.isfinite(out).all()
    F.l1_loss(out,target).backward()
    for name,branch in [('dge',model.dge),('gre',model.gre),*model.adapters.items(),*model.geometry_routes.items()]:
        grads=[p.grad for p in branch.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads),name
        assert sum(g.abs().sum().item() for g in grads)>0,name
    opt.step()
    print('PASS: six-stage forward/backward, first-step gradients in all geometry branches, Adam update')
    # Check exact RGB fallback against the pre-change source, not just this revision.
    oldsource=subprocess.check_output(['git','show','d84240a:basicsr/archs/mambairv2_arch.py'],cwd=ROOT).decode()
    old=load_isolated(oldsource)
    if not a.cpu_reference:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        old.Selective_Scan.__init__.__globals__['selective_scan_fn']=selective_scan_fn
    pure=old.MambaIRv2(**small).to(device)
    pure.load_state_dict({k:v for k,v in model.state_dict().items() if k in pure.state_dict()},strict=True)
    for m in model.adapters.values(): m.alpha.data.zero_()
    for m in model.geometry_routes.values(): m.rho.data.zero_()
    with torch.no_grad():
        torch.manual_seed(123); expected=pure(rgb)
        torch.manual_seed(123); actual=model(rgb,depth)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    print('PASS: alpha=rho=0 gives bit-identical d84240a RGB baseline output under the same seed')
    # Verify checkpoint completeness and repeatability (strict-load final weights).
    other=grs.GRSMambaIRv2(**small).to(device)
    other.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        torch.manual_seed(123); restored=other(rgb,depth)
        tiny=model(rgb[...,:1,:2],depth[...,:1,:2])
    torch.testing.assert_close(restored,actual,rtol=0,atol=0)
    assert tiny.shape==(1,3,4,8) and torch.isfinite(tiny).all()
    confidence=model.gre(rgb,torch.zeros_like(depth))
    assert torch.isfinite(confidence).all() and confidence.min()>=0 and confidence.max()<=1
    try: model(rgb,depth[...,:-1,:])
    except ValueError: pass
    else: raise AssertionError('Depth mismatch was accepted')
    check_partition()
    check_dataset_locally()
    if a.check_data:
        if a.cpu_reference:
            raise ValueError('Run --check-data in the actual server CUDA environment.')
        from basicsr.data import build_dataset
        testcfg=yaml.safe_load((ROOT/'options/test/mambairv2/test_GRS_MambaSR_x4.yml').read_text())
        for phase,dsopt in list(cfg['datasets'].items())+list(testcfg['datasets'].items()):
            dsopt.update(phase=phase.split('_')[0],scale=4)
            ds=build_dataset(dsopt)
            for i in sorted({0,len(ds)//2,len(ds)-1}):
                v=ds[i]
                assert v['depth'].shape[-2:]==v['lq'].shape[-2:]
            print(f"PASS: {dsopt['name']} all filenames paired; first/middle/last decoded ({len(ds)} images)")
    print('ALL CHECKS PASSED ('+('CPU reference; CUDA/server data not exercised' if a.cpu_reference else 'CUDA')+')')


if __name__=='__main__':
    main()
