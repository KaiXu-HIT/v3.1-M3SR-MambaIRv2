"""Paired three-seed baseline/GRS inference; report dataset means and sample std.

Each worker runs in a fresh process. Seeds are reset AFTER network construction
and before each dataset, so extra geometry initialization cannot shift Gumbel RNG.
Both models use the original MambaIRv2 partition/overlap/Y-channel metric pipeline.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(config, output):
    import torch
    from basicsr.data import build_dataset, build_dataloader
    from basicsr.models import build_model
    from basicsr.utils import set_random_seed, make_exp_dirs
    from basicsr.utils.options import parse_options
    sys.argv = [sys.argv[0], '-opt', str(config)]
    opt, _ = parse_options(str(ROOT), is_train=False)
    make_exp_dirs(opt)
    # Same deterministic cuDNN setting for both models; Gumbel remains stochastic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(opt)
    metrics = {}
    for _, dsopt in sorted(opt['datasets'].items()):
        dataset = build_dataset(dsopt)
        # Explicit matched image order in both RGB-only and paired datasets.
        dataset.paths.sort(key=lambda item: item['lq_path'])
        loader = build_dataloader(dataset, dsopt, num_gpu=1, dist=False,
                                  sampler=None, seed=opt['manual_seed'])
        set_random_seed(opt['manual_seed'])
        model.validation(loader, current_iter=opt['name'], tb_logger=None, save_img=False)
        metrics[dsopt['name']] = dict(model.metric_results)
    Path(output).write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='options/test/mambairv2/test_GRS_RGB_reference_x4.yml')
    p.add_argument('--grs', default='options/test/mambairv2/test_GRS_MambaSR_x4.yml')
    p.add_argument('--baseline-checkpoint')
    p.add_argument('--grs-checkpoint')
    p.add_argument('--seeds', nargs=3, type=int, default=[10, 11, 12])
    p.add_argument('--output', default='results/grs_comparison')
    p.add_argument('--worker', nargs=2, metavar=('CONFIG', 'JSON'))
    a = p.parse_args()
    if a.worker:
        worker(*a.worker)
        return
    if len(set(a.seeds)) != 3:
        p.error('Use three distinct seeds, paired identically between models.')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    configs = {}
    for label, path in [('baseline', a.baseline), ('grs', a.grs)]:
        config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        checkpoint = getattr(a, label + '_checkpoint') or config['path']['pretrain_network_g']
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config['path']['pretrain_network_g'] = str(checkpoint)
        config['path']['strict_load_g'] = True
        configs[label] = config
    if configs['baseline']['val']['metrics'] != configs['grs']['val']['metrics']:
        raise ValueError('Both models must use exactly the same metrics.')
    for key, bd in configs['baseline']['datasets'].items():
        gd = configs['grs']['datasets'][key]
        for field in ('name', 'dataroot_gt', 'dataroot_lq', 'filename_tmpl'):
            if bd[field] != gd[field]:
                raise ValueError(f'Unmatched evaluation input: {key}/{field}')
    runs = {'baseline': [], 'grs': []}
    for seed in a.seeds:
        for label in runs:
            config = copy.deepcopy(configs[label])
            config['manual_seed'] = seed
            config['name'] = f'GRS_fair_{label}_seed{seed}'
            cfgpath, result = out/f'{label}_{seed}.yml', out/f'{label}_{seed}.json'
            cfgpath.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--worker', str(cfgpath), str(result)], cwd=ROOT, check=True)
            runs[label].append(json.loads(result.read_text(encoding='utf-8')))
    report = {'seeds': a.seeds, 'std_ddof': 1, 'runs': runs, 'summary': {}}
    lines = ['# GRS-MambaSR paired evaluation', '',
             'Three distinct seeds, paired between models; sample standard deviation (ddof=1).', '',
             '| Dataset | RGB PSNR | GRS PSNR | Delta PSNR | RGB SSIM | GRS SSIM |',
             '|---|---:|---:|---:|---:|---:|']
    deltas = []
    for ds in runs['baseline'][0]:
        summary = {}
        for label in runs:
            summary[label] = {m: {'mean': statistics.mean(x[ds][m] for x in runs[label]),
                                  'std': statistics.stdev(x[ds][m] for x in runs[label])}
                              for m in ('psnr', 'ssim')}
        delta = summary['grs']['psnr']['mean'] - summary['baseline']['psnr']['mean']
        summary['delta_psnr'] = delta
        report['summary'][ds] = summary
        deltas.append(delta)
        def fmt(label, metric):
            v = summary[label][metric]
            return f"{v['mean']:.4f} +/- {v['std']:.4f}"
        lines.append(f"| {ds} | {fmt('baseline','psnr')} | {fmt('grs','psnr')} | {delta:+.4f} | {fmt('baseline','ssim')} | {fmt('grs','ssim')} |")
    report['average_delta_psnr'] = statistics.mean(deltas)
    report['nondecreasing_datasets'] = sum(d >= 0 for d in deltas)
    report['minimum_effective'] = report['average_delta_psnr'] >= 0.03 and report['nondecreasing_datasets'] >= 4
    lines.extend(['', f"Five-set average delta: {report['average_delta_psnr']:+.4f} dB.",
                  f"Nondecreasing datasets: {report['nondecreasing_datasets']}/5.",
                  f"Minimum effective criterion: {report['minimum_effective']}."])
    (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
