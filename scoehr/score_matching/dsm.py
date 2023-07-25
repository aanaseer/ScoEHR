"""Implements the denoising score matching objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingScoreMatching(nn.Module):
    def __init__(self, sde, score_net, T=1.0, padding_required=False):
        super().__init__()
        self.sde = sde
        self.score_net = score_net
        self.T = T
        self.padding_required = padding_required

    def pad_to(self, x, stride):
        h, w = x.shape[-2:]
        if w % stride > 0:
            new_w = w + stride - w % stride
        else:
            new_w = w
        lh, uh = int((h - h) / 2), int(h - h) - int((h - h) / 2)
        lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
        pads = (lw, uw, lh, uh)

        # zero-padding by default.
        # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
        out = F.pad(x, pads, "constant", 0)

        return out, pads

    def unpad(self, x, pad):
        if pad[0] + pad[1] > 0:
            x = x[:, pad[0] : -pad[1]]
        return x

    @torch.enable_grad()
    def loss_fn(self, x):
        t = torch.linspace(1 / x.size(0) + 1e-3, 1 - 1 / x.size(0), x.size(0)) + (
            1 - 2 * torch.rand(x.size(0))
        ) * (1 / x.size(0))
        t = t.view(-1, 1).to(x)

        idx = torch.randperm(t.nelement())
        t = t.view(-1)[idx].view(t.size())

        perturbed_x, noise, std, g = self.sde.sample(x, t)

        if self.padding_required:
            perturbed_x, pads = self.pad_to(perturbed_x, 784)
        score_predictions = self.score_net(perturbed_x, t)
        if self.padding_required:
            score_predictions = self.unpad(score_predictions, pads)

        return ((score_predictions + noise / std) ** 2).view(x.size(0), -1).sum(
            1, keepdim=False
        ) * (std**2).squeeze()
