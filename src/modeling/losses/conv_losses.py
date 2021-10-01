from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from kornia.losses import total_variation
from torch import nn

from src.modeling.model_arch.conv_models import SegAct


class DenoiseLoss(nn.Module):
    def __init__(
        self,
        batch_size: int = 128,
        num_classes: int = 1,
        num_sites: int = 3,
        crop_size: int = 128,
        target_loss_weight: float = 0.01,
        l1_loss_weight: float = 10.0,
        var_loss_weight: float = 0.001,
        predict_mask: bool = True,
    ):

        super().__init__()
        self.crop_size = crop_size

        regularize_channels = 1 if predict_mask else num_sites
        self.zero_mask = nn.Parameter(
            data=torch.zeros(batch_size, regularize_channels, crop_size, crop_size),
            requires_grad=False,
        )
        self.target_loss_weight = target_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.var_loss_weight = var_loss_weight
        self.seg_act = SegAct(predict_mask=predict_mask)
        self.predict_mask = predict_mask

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # velocity (outputs[:, :, 1:,] - outputs[:, :, :-1, ])
        # acc nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(outputs) - outputs

        if self.predict_mask:
            # only apply mask
            regularize_logits = logits[:, 1:2]
        else:
            regularize_logits = logits

        if self.l1_loss_weight == 0.0:
            l1_loss = 0.0
        else:
            l1_loss = F.smooth_l1_loss(regularize_logits, target=self.zero_mask)
        if self.var_loss_weight == 0.0:
            var_loss = 0.0
        else:
            var_loss = total_variation(img=regularize_logits).mean()

        if self.predict_mask:
            target_loss = F.binary_cross_entropy_with_logits(
                input=self.seg_act(logits, is_act=False), target=targets
            )
        else:
            target_loss = F.mse_loss(
                input=self.seg_act(logits, is_act=True), target=targets
            )
        return (
            self.target_loss_weight * target_loss
            + self.l1_loss_weight * l1_loss
            + self.var_loss_weight * var_loss
        )
