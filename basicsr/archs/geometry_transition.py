"""GTSS specification sections 10/13/14/16: one scalar per modulated ASSM."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GeometryTransitionController(nn.Module):
    """Gather scalar geometry on the ASSM's own semantic path, then difference.

    This contains no feature encoder, learned threshold, or channel projection.
    beta=0.5*sigmoid(theta), initialized to 0.02, broadcasts over hidden channels.
    """
    beta_max = 0.5
    q_max = 0.25

    def __init__(self):
        super().__init__()
        self.beta_logit = nn.Parameter(torch.tensor(math.log(0.02 / (0.5 - 0.02))))
        self.gate_mean = None

    @property
    def beta(self):
        return self.beta_max * torch.sigmoid(self.beta_logit)

    def forward(self, depth, confidence, sort_indices):
        if depth.ndim != 4 or depth.shape[1] != 1 or depth.shape != confidence.shape:
            raise ValueError('GTSS depth/confidence must be matching [B,1,H,W] maps.')
        d, c = depth.flatten(1), confidence.flatten(1)
        if d.shape != sort_indices.shape:
            raise ValueError('GTSS maps must match the ASSM token count exactly.')
        # Use this ASSM's exact permutation, including within-class ordering.
        ds = torch.gather(d, 1, sort_indices)
        cs = torch.gather(c, 1, sort_indices)
        q = (ds[:, 1:] - ds[:, :-1]).abs().clamp(0, self.q_max) / self.q_max
        reliability = torch.minimum(cs[:, 1:], cs[:, :-1])
        # There is no previous token for the first transition; G_1 is exactly 0.
        gate = F.pad(reliability * q, (1, 0)).unsqueeze(1)
        self.gate_mean = gate.detach().mean()
        return gate
