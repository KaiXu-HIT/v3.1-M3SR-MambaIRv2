"""GTSS-MambaSR v3.1: scalar depth -> semantic-path transition -> dts only.

ASSB 2/4/6 each enable six independently bounded controllers. RGB feature
residuals, semantic route logits, B/C and the CUDA kernel are never modified
by the direct depth path. Legacy GRS exists only for historical evaluation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.archs.mambairv2_arch import MambaIRv2
from basicsr.archs.geometry_transition import GeometryTransitionController
from basicsr.utils.registry import ARCH_REGISTRY


def robust_normalize(x, eps=1e-6):
    """Per-image P2/P98 normalization, independently for RGB/depth edge maps."""
    flat = x.float().flatten(1)
    low = torch.quantile(flat, 0.02, dim=1).view(-1, 1, 1, 1)
    high = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1, 1)
    return ((x - low) / (high - low + eps)).clamp(0, 1).to(x.dtype)


class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer('kernels', torch.stack([k, k.t()]).unsqueeze(1))

    def forward(self, x):
        # Replicate padding avoids artificial depth edges on constant images.
        g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'),
                     self.kernels.to(x))
        return (g.square().sum(1, keepdim=True) + 1e-6).sqrt()


class GeometryReliabilityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelGradient()
        self.register_buffer('luma', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.GELU(),
            nn.Conv2d(8, 1, 1))

    def forward(self, rgb, depth_edge):
        # Use raw [0,1] RGB, before backbone mean subtraction.
        er = robust_normalize(self.sobel((rgb * self.luma.to(rgb)).sum(1, keepdim=True)))
        ed = robust_normalize(depth_edge)
        z = torch.cat([er, ed, (er-ed).abs(), er*ed], 1)
        return torch.sigmoid(self.net(z))


@ARCH_REGISTRY.register()
class GTSSMambaIRv2(MambaIRv2):
    def __init__(self, **kwargs):
        defaults = dict(in_chans=3, upscale=4, upsampler='pixelshuffle',
                        embed_dim=174, depths=(6,)*6, num_heads=(6,)*6,
                        d_state=16, inner_rank=64, num_tokens=128)
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        if kwargs['in_chans'] != 3 or kwargs['upscale'] != 4 or kwargs['upsampler'] != 'pixelshuffle':
            raise ValueError('GTSS requires classical RGB x4 with the baseline pixelshuffle head.')
        if len(kwargs['depths']) != 6:
            raise ValueError('GTSS uses six stages, with geometry only at ASSB 2/4/6.')
        super().__init__(**kwargs)
        self.depth_sobel = SobelGradient()
        self.gre = GeometryReliabilityEstimator()
        # Attach AFTER baseline initialization; each ASSM has its own beta.
        for stage in (2, 4, 6):
            for layer in self.layers[stage-1].residual_group.layers:
                layer.assm.geometry_controller = GeometryTransitionController()
        self.geometry_stats = {}

    def forward_features(self, x, params):
        size = x.shape[-2:]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        for stage, layer in enumerate(self.layers, start=1):
            # Do not compute a transition map here: sort order is per ASSM.
            stage_params = dict(attn_mask=params['attn_mask'], rpi_sa=params['rpi_sa'])
            if stage in (2, 4, 6):
                stage_params.update(depth=params['depth'], confidence=params['confidence'],
                                    enable_gtss=True)
            x = layer(x, size, stage_params)
        return self.patch_unembed(self.norm(x), size)

    @staticmethod
    def _pad_to_size(x, h, w):
        # Same symmetric extension as RGB baseline; support tiny inputs too.
        while x.shape[-2] < h:
            x = torch.cat([x, x.flip([2])], 2)
        while x.shape[-1] < w:
            x = torch.cat([x, x.flip([3])], 3)
        return x[..., :h, :w]

    def forward(self, rgb, depth):
        if rgb.ndim != 4 or depth.ndim != 4 or rgb.shape[1] != 3 or depth.shape[1] != 1:
            raise ValueError('Expected [B,3,H,W] RGB and [B,1,H,W] normalized depth.')
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError('RGB/depth must be aligned; implicit depth resizing is forbidden.')
        h0, w0 = rgb.shape[-2:]
        h = (h0 + self.window_size - 1) // self.window_size * self.window_size
        w = (w0 + self.window_size - 1) // self.window_size * self.window_size
        # Full-image depth normalization was done by the unchanged dataset.
        # Sobel edges ONLY estimate reliability; they never form the transition.
        confidence = self.gre(rgb, self.depth_sobel(depth))
        params = dict(attn_mask=self.calculate_mask([h,w]).to(rgb.device),
                      rpi_sa=self.relative_position_index_SA,
                      depth=self._pad_to_size(depth, h, w),
                      confidence=self._pad_to_size(confidence, h, w))
        mean = self.mean.to(rgb)
        x = (self._pad_to_size(rgb, h, w) - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x, params)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x)) / self.img_range + mean
        # Detach diagnostics: logging must not keep the scan autograd graph.
        self.geometry_stats = dict(confidence_mean=confidence.detach().mean(),
                                   confidence_std=confidence.detach().std(unbiased=False))
        gates = []
        for stage in (2,4,6):
            for j, layer in enumerate(self.layers[stage-1].residual_group.layers, start=1):
                controller = layer.assm.geometry_controller
                self.geometry_stats[f'beta_{stage}_{j}'] = controller.beta.detach()
                self.geometry_stats[f'gate_mean_{stage}_{j}'] = controller.gate_mean
                gates.append(controller.gate_mean)
        self.geometry_stats['transition_gate_mean'] = torch.stack(gates).mean()
        return x[..., :h0*self.upscale, :w0*self.upscale]
