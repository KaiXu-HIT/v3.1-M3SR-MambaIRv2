"""GTSS mechanism/compatibility checks; --cpu-reference does not test CUDA.

Default runs the real installed BasicSR/Mamba CUDA path. --check-data also
checks all filenames and decodes first/middle/last samples in each dataset.
"""
import argparse
import ast
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.grs.check_grs import load_isolated, check_dataset_locally, check_partition


def git_source(path, ref='d84240a'):
    return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT).decode('utf-8')


def method(path,name):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
    return next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)


def check_gate(Controller,device):
    ctrl=Controller().to(device)
    d=torch.tensor([0.,.1,.8,.85],device=device).reshape(1,1,1,4)
    c=torch.tensor([.2,.4,.8,1.],device=device).reshape_as(d)
    index=torch.tensor([[2,3,1,0]],device=device)
    expected=torch.tensor([[[0.,.16,.4,.08]]],device=device)
    torch.testing.assert_close(ctrl(d,c,index),expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(ctrl(d,torch.zeros_like(c),index),torch.zeros_like(expected),rtol=0,atol=0)
    torch.testing.assert_close(ctrl(torch.ones_like(d),c,index),torch.zeros_like(expected),rtol=0,atol=0)
    assert ctrl(d[...,:1],c[...,:1],torch.zeros(1,1,dtype=torch.long,device=device)).item()==0
    assert abs(ctrl.beta.item()-.02)<1e-7
    for theta in (-12.,0.,12.):
        ctrl.beta_logit.data.fill_(theta)
        assert 0<ctrl.beta.item()<.5
    try: ctrl(d,c,index[:,:2])
    except ValueError: pass
    else: raise AssertionError('Invalid permutation length accepted')
    print('PASS: hand-computed sorted gate, endpoint min, q_max clipping, first/constant/zero-confidence transitions, beta bounds')


def check_scan(base,device):
    scan=base.Selective_Scan(d_model=6,d_state=2,expand=1).to(device)
    x=torch.randn(2,5,6,device=device)
    prompt=torch.randn(2,5,2,device=device)
    g=torch.tensor([[[0.,.1,.4,.9,.3]],[[0.,.8,.2,.5,1.]]],device=device)
    beta=torch.tensor(.02,device=device)
    recorded=[]
    original=scan.selective_scan
    def capture(*args,**kwargs):
        recorded.append(([v.detach().clone() for v in args],
                         {k:v.detach().clone() if torch.is_tensor(v) else v for k,v in kwargs.items()}))
        return original(*args,**kwargs)
    scan.selective_scan=capture
    scan(x,prompt)
    scan(x,prompt,geometry_transition=g,beta=beta)
    a,ak=recorded[0];b,bk=recorded[1]
    expected=beta*g.expand(2,6,5)
    torch.testing.assert_close(b[1]-a[1],expected,atol=1e-6,rtol=1e-5)
    for i in (0,2,3,4,5):
        torch.testing.assert_close(a[i],b[i],rtol=0,atol=0)
    torch.testing.assert_close(ak['delta_bias'],bk['delta_bias'],rtol=0,atol=0)
    assert ak['delta_softplus'] is True and bk['delta_softplus'] is True
    assert torch.equal(a[1][...,0],b[1][...,0])
    de0=F.softplus(a[1]+ak['delta_bias'][None,:,None])
    de1=F.softplus(b[1]+bk['delta_bias'][None,:,None])
    assert (de1>=de0).all() and (a[2]<0).all()
    retention0=torch.exp(de0.unsqueeze(-1)*a[2][None,:,None,:])
    retention1=torch.exp(de1.unsqueeze(-1)*b[2][None,:,None,:])
    assert (retention1<=retention0).all()
    print('PASS: only raw dts changes; batch/channel broadcasting, B/C/A/input/bias unchanged, softplus and decay direction correct')


def check_assm_routing(model,device):
    layer=model.layers[1].residual_group.layers[0]
    assm=layer.assm
    x=torch.rand(2,16,model.embed_dim,device=device)
    depth=torch.rand(2,1,4,4,device=device)
    confidence=torch.rand_like(depth)
    indices=[]; scan_inputs=[]
    h1=assm.geometry_controller.register_forward_pre_hook(
        lambda m,args: indices.append(args[2].detach().clone()))
    h2=assm.selectiveScan.register_forward_pre_hook(
        lambda m,args: scan_inputs.append(tuple(v.detach().clone() for v in args)))
    with torch.no_grad():
        torch.manual_seed(77)
        first=assm(x,(4,4),layer.embeddingA,depth,confidence,True)
        torch.manual_seed(77)
        second=assm(x,(4,4),layer.embeddingA,depth.flip([-1]),confidence,True)
    h1.remove();h2.remove()
    assert torch.equal(indices[0],indices[1])
    for a,b in zip(scan_inputs[0],scan_inputs[1]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(first,second)
    print('PASS: changing depth cannot directly change ASSM route/permutation/prompt/scan input, but does affect scan output')


def check_evaluation_summary():
    from scripts.gtss.evaluate_repeated import summarize
    runs={label:[] for label in ('baseline','gtss','grs')}
    for seed in (10,11,12):
        for label,delta in [('baseline',0),('gtss',.06),('grs',-.009)]:
            runs[label].append({name:dict(psnr=30+seed*.01+delta,ssim=.9+seed*.0001)
                                for name in ('Set5','Set14','B100','Urban100','Manga109')})
    report,_=summarize(runs,[10,11,12])
    assert abs(report['average_delta_vs_rgb']['gtss']-.06)<1e-12
    assert abs(report['summary']['Set5']['baseline']['psnr']['std']-.01)<1e-12
    assert report['gtss_criteria']['average_target_met']
    assert not report['gtss_criteria']['urban100_target_met']
    print('PASS: three-model mean/sample-std/RGB-referenced criteria (synthetic software fixtures, not performance results)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-reference',action='store_true')
    parser.add_argument('--check-data',action='store_true')
    args=parser.parse_args()
    if args.cpu_reference and args.check_data:
        parser.error('--check-data requires the original server CUDA environment.')
    torch.set_num_threads(2);torch.manual_seed(10);random.seed(10)
    if args.cpu_reference:
        base=load_isolated((ROOT/'basicsr/archs/mambairv2_arch.py').read_text())
        ctl=load_isolated((ROOT/'basicsr/archs/geometry_transition.py').read_text())
        gtss=load_isolated((ROOT/'basicsr/archs/gtss_mambairv2_arch.py').read_text(),
            dict(MambaIRv2=base.MambaIRv2,GeometryTransitionController=ctl.GeometryTransitionController))
        device='cpu'
    else:
        if not torch.cuda.is_available():
            parser.error('CUDA unavailable; use --cpu-reference only for local mechanism checks.')
        from basicsr.archs import mambairv2_arch as base, gtss_mambairv2_arch as gtss
        from basicsr.archs import geometry_transition as ctl
        device='cuda'
    check_gate(ctl.GeometryTransitionController,device)
    check_scan(base,device)
    cfg=yaml.safe_load((ROOT/'options/train/mambairv2/train_GTSS_MambaSR_x4.yml').read_text())
    prior=yaml.safe_load((ROOT/'options/train/mambairv2/train_GRS_MambaSR_x4.yml').read_text())
    assert cfg['datasets']==prior['datasets'] and cfg['train']==prior['train']
    assert cfg['path']['pretrain_network_g'] is None and cfg['train']['total_iter']==500000
    testcfg=yaml.safe_load((ROOT/'options/test/mambairv2/test_GTSS_MambaSR_x4.yml').read_text())
    oldtest=yaml.safe_load((ROOT/'options/test/mambairv2/test_GRS_MambaSR_x4.yml').read_text())
    assert testcfg['datasets']==oldtest['datasets'] and testcfg['val']==oldtest['val']
    netcfg=copy.deepcopy(cfg['network_g']);netcfg.pop('type')
    full=gtss.GTSSMambaIRv2(**netcfg).to(device)
    pure=base.MambaIRv2(**netcfg)
    count=lambda m:sum(p.numel() for p in m.parameters())
    n0,n1=count(pure),count(full)
    assert n1-n0==1779
    controllers=[]
    for i,stage in enumerate(full.layers,1):
        for layer in stage.residual_group.layers:
            ctrl=layer.assm.geometry_controller
            if i in (2,4,6):
                assert ctrl is not None and abs(ctrl.beta.item()-.02)<1e-7
                controllers.append(ctrl)
            else:
                assert ctrl is None
    assert len(controllers)==18 and len({id(c.beta_logit) for c in controllers})==18
    assert not any(hasattr(full,name) for name in ('dge','adapters','geometry_routes'))
    extra=set(full.state_dict())-set(pure.state_dict())
    assert all(k.startswith(('gre.','depth_sobel.')) or k.endswith('geometry_controller.beta_logit') for k in extra)
    print(json.dumps(dict(baseline_params=n0,gtss_params=n1,added=n1-n0,added_percent=100*(n1-n0)/n0)))
    side=16 if args.cpu_reference else 48
    with torch.no_grad():
        output=full(torch.rand(1,3,side,side,device=device),torch.rand(1,1,side,side,device=device))
    assert output.shape==(1,3,4*side,4*side) and torch.isfinite(output).all()
    assert len([k for k in full.geometry_stats if k.startswith('beta_')])==18
    print('PASS: full 174-channel/36-layer forward, exactly 18 independent betas at ASSB 2/4/6, no DGE/RAGA/GCR')
    del full,pure,output
    small=dict(img_size=8,embed_dim=12,d_state=4,depths=[6]*6,num_heads=[3]*6,
               window_size=4,inner_rank=4,num_tokens=8,convffn_kernel_size=5,
               mlp_ratio=2.,upscale=4,upsampler='pixelshuffle')
    model=gtss.GTSSMambaIRv2(**small).to(device)
    check_assm_routing(model,device)
    rgb=torch.rand(2,3,5,7,device=device);depth=torch.rand(2,1,5,7,device=device)
    # Run the production optimize_parameters method, including diagnostic logging.
    fn=method('basicsr/models/gtss_mambairv2_model.py','optimize_parameters')
    env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<model optimize>','exec'),env)
    obj=types.SimpleNamespace(net_g=model,lq=rgb,depth=depth,gt=torch.rand(2,3,20,28,device=device),
        optimizer_g=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.9,.99)),
        cri_pix=nn.L1Loss(),cri_perceptual=None,ema_decay=0,
        get_bare_model=lambda net:net,reduce_loss_dict=lambda data:{k:v.detach().item() for k,v in data.items()})
    env['optimize_parameters'](obj,1)
    assert len([k for k in obj.log_dict if k.startswith('beta_')])==18
    for name,p in model.named_parameters():
        if name.startswith('gre.') or name.endswith('geometry_controller.beta_logit'):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    print('PASS: real L1 optimize method, first-step GRE/all-18-beta gradients, Adam and diagnostics')
    old=load_isolated(git_source('basicsr/archs/mambairv2_arch.py'))
    if not args.cpu_reference:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        old.Selective_Scan.__init__.__globals__['selective_scan_fn']=selective_scan_fn
    original=old.MambaIRv2(**small).to(device)
    original.load_state_dict({k:v for k,v in model.state_dict().items() if k in original.state_dict()},strict=True)
    current_rgb=base.MambaIRv2(**small).to(device)
    current_rgb.load_state_dict(original.state_dict(),strict=True)
    with torch.no_grad():
        torch.manual_seed(42); expected=original(rgb)
        torch.manual_seed(42); actual=model(rgb,torch.ones_like(depth)*.4)
        torch.manual_seed(42); unchanged=current_rgb(rgb)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(unchanged,expected,rtol=0,atol=0)
    print('PASS: constant depth gives bit-identical d84240a baseline; active RGB-only path also unchanged')
    payload=io.BytesIO();torch.save(model.state_dict(),payload);payload.seek(0)
    restored=gtss.GTSSMambaIRv2(**small).to(device)
    restored.load_state_dict(torch.load(payload,map_location=device),strict=True)
    with torch.no_grad():
        torch.manual_seed(99);a=model(rgb,depth)
        torch.manual_seed(99);b=restored(rgb,depth)
        tiny=restored(rgb[:1,:,:1,:2],depth[:1,:,:1,:2])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert tiny.shape==(1,3,4,8) and torch.isfinite(tiny).all()
    # Old/new wrappers have exactly the same test-method AST; reuse its coverage.
    assert ast.dump(method('basicsr/models/gtss_mambairv2_model.py','test'))==ast.dump(method('basicsr/models/grs_mambairv2_model.py','test'))
    check_partition();check_dataset_locally();check_evaluation_summary()
    # The legacy backbone is a frozen source snapshot except for registration.
    frozen=(ROOT/'basicsr/archs/legacy_grs/mambairv2_v30.py').read_text()
    v30=git_source('basicsr/archs/mambairv2_arch.py','697f97a')
    assert frozen.split('\n',1)[1].rstrip()==v30.replace('@ARCH_REGISTRY.register()\nclass MambaIRv2','class MambaIRv2',1).rstrip()
    print('PASS: strict checkpoint round-trip, tiny images and frozen v3.0 source compatibility')
    if args.check_data:
        from basicsr.data import build_dataset
        for phase,dsopt in list(cfg['datasets'].items())+list(testcfg['datasets'].items()):
            dsopt.update(phase=phase.split('_')[0],scale=4)
            ds=build_dataset(dsopt)
            for i in sorted({0,len(ds)//2,len(ds)-1}):
                v=ds[i]
                assert v['depth'].shape[-2:]==v['lq'].shape[-2:]
            print(f"PASS: {dsopt['name']} all filenames paired; first/middle/last decoded ({len(ds)} samples)")
    print('ALL GTSS CHECKS PASSED ('+('CPU reference; not CUDA/server-data validation' if args.cpu_reference else 'CUDA')+')')


if __name__=='__main__':
    main()
