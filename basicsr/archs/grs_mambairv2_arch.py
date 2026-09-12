"""GRS-MambaSR: asymmetric geometry guidance on the d84240a RGB backbone.

Only ASSB 2/4/6 receive RAGA and GCR. Each stage shares one geometry
projection/rho across its six ASSMs; all RGB routing dictionaries stay separate.
Depth must already be P2/P98-normalized using the complete LR depth image.
"""
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.archs.mambairv2_arch import MambaIRv2
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


class DepthGeometryEncoder(nn.Module):
    def __init__(self, dim=174):
        super().__init__()
        self.sobel = SobelGradient()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1, groups=48),
            nn.Conv2d(48, 48, 1), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1, groups=48),
            nn.Conv2d(48, 48, 1), nn.Conv2d(48, dim, 3, padding=1))
        # Explicit Kaiming initialization for the newly introduced encoder.
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, depth):
        edge = self.sobel(depth)
        return self.encoder(torch.cat([depth, edge], 1)), edge


class GeometryReliabilityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelGradient()
        self.register_buffer('luma', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, 1))

    def forward(self, rgb, depth_edge):
        # Use raw [0,1] RGB, before backbone mean subtraction.
        er = robust_normalize(self.sobel((rgb * self.luma.to(rgb)).sum(1, keepdim=True)))
        ed = robust_normalize(depth_edge)
        z = torch.cat([er, ed, (er-ed).abs(), er*ed], 1)
        return torch.sigmoid(self.net(z))


class ReliabilityAwareGeometryAdapter(nn.Module):
    def __init__(self, dim=174):
        super().__init__()
        self.depth_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1))
        # The specification leaves gate bottleneck width open: use C//16.
        hidden = max(dim // 16, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, dim, 1), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.tensor(0.05))
        nn.init.normal_(self.depth_proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.depth_proj[-1].bias)
        self.residual_abs = None

    def forward(self, rgb_feat, depth_feat, confidence):
        residual = self.alpha * confidence * self.channel_gate(rgb_feat) * self.depth_proj(depth_feat)
        self.residual_abs = residual.detach().abs().mean()
        return rgb_feat + residual


class GeometryConditionedRouting(nn.Module):
    def __init__(self, dim=174, num_tokens=128):
        super().__init__()
        # R_D = W_D D_t from the specification; one projection per ASSB.
        self.depth_route = nn.Linear(dim, num_tokens)
        self.rho = nn.Parameter(torch.tensor(0.05))
        nn.init.normal_(self.depth_route.weight, std=1e-3)
        nn.init.zeros_(self.depth_route.bias)

    def forward(self, depth_feat, confidence):
        dt = depth_feat.flatten(2).transpose(1, 2)
        ct = confidence.flatten(2).transpose(1, 2)
        return self.rho * ct * self.depth_route(dt)


@ARCH_REGISTRY.register()
class GRSMambaIRv2(MambaIRv2):
    def __init__(self, **kwargs):
        if kwargs.get('in_chans', 3) != 3 or kwargs.get('upscale', 4) != 4:
            raise ValueError('GRS-MambaSR requires RGB x4.')
        if kwargs.get('upsampler', 'pixelshuffle') != 'pixelshuffle':
            raise ValueError('GRS-MambaSR uses the baseline pixelshuffle head.')
        kwargs.setdefault('in_chans', 3)
        kwargs.setdefault('upscale', 4)
        kwargs.setdefault('upsampler', 'pixelshuffle')
        kwargs.setdefault('embed_dim', 174)
        kwargs.setdefault('depths', (6,)*6)
        kwargs.setdefault('num_heads', (6,)*6)
        kwargs.setdefault('d_state', 16)
        kwargs.setdefault('inner_rank', 64)
        kwargs.setdefault('num_tokens', 128)
        if len(kwargs['depths']) != 6:
            raise ValueError('GRS-MambaSR requires six ASSB stages.')
        super().__init__(**kwargs)
        self.dge = DepthGeometryEncoder(self.embed_dim)
        self.gre = GeometryReliabilityEstimator()
        self.adapters = nn.ModuleDict({str(i): ReliabilityAwareGeometryAdapter(self.embed_dim)
                                       for i in (2, 4, 6)})
        self.geometry_routes = nn.ModuleDict({str(i): GeometryConditionedRouting(
            self.embed_dim, kwargs['num_tokens']) for i in (2, 4, 6)})
        # New modules are initialized AFTER the baseline's self.apply(), so
        # small final projections cannot be overwritten by baseline init.
        self.geometry_stats = {}

    def forward_features(self, x, params):
        size = x.shape[-2:]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        fd, confidence = params['depth_feat'], params['confidence']
        for i, layer in enumerate(self.layers, start=1):
            stage_params = dict(params)
            if str(i) in self.adapters:
                feature = self.patch_unembed(x, size)
                feature = self.adapters[str(i)](feature, fd, confidence)
                # Flatten directly: do not apply PatchEmbed LayerNorm twice.
                x = feature.flatten(2).transpose(1, 2)
                stage_params['geometry_route_bias'] = self.geometry_routes[str(i)](fd, confidence)
            x = layer(x, size, stage_params)
        return self.patch_unembed(self.norm(x), size)

    @staticmethod
    def _pad_to_size(x, h, w):
        # Identical symmetric extension to baseline for ordinary images;
        # repeated doubling also handles LR images smaller than half a window.
        while x.shape[-2] < h:
            x = torch.cat([x, x.flip([2])], 2)
        while x.shape[-1] < w:
            x = torch.cat([x, x.flip([3])], 3)
        return x[..., :h, :w]

    def forward(self, rgb, depth):
        if rgb.ndim != 4 or depth.ndim != 4 or rgb.shape[1] != 3 or depth.shape[1] != 1:
            raise ValueError('Expected BCHW RGB (3 channels) and depth (1 channel).')
        if rgb.shape[0] != depth.shape[0] or rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError('RGB and depth must be aligned; resizing depth silently is forbidden.')
        h0, w0 = rgb.shape[-2:]
        h = (h0 + self.window_size - 1) // self.window_size * self.window_size
        w = (w0 + self.window_size - 1) // self.window_size * self.window_size
        # Geometry is estimated before padding to keep robust edge statistics
        # independent of artificial window padding. Extend all maps identically.
        fd, ed = self.dge(depth)
        confidence = self.gre(rgb, ed)
        self.geometry_stats = {
            'confidence_mean': confidence.detach().mean(),
            'confidence_std': confidence.detach().std(unbiased=False)}
        params = {
            'attn_mask': self.calculate_mask([h, w]).to(rgb.device),
            'rpi_sa': self.relative_position_index_SA,
            'depth_feat': self._pad_to_size(fd, h, w),
            'confidence': self._pad_to_size(confidence, h, w)}
        mean = self.mean.to(rgb)
        x = (self._pad_to_size(rgb, h, w) - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x, params)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x)) / self.img_range + mean
        for key in self.adapters:
            self.geometry_stats['alpha_' + key] = self.adapters[key].alpha.detach().clone()
            self.geometry_stats['rho_' + key] = self.geometry_routes[key].rho.detach().clone()
        self.geometry_stats['depth_residual_abs_mean'] = torch.stack(
            [a.residual_abs for a in self.adapters.values()]).mean()
        return x[..., :h0 * self.upscale, :w0 * self.upscale]
