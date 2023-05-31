import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.global_projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.global_projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        local_similarity,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {
            "instance_loss": losses.cross_entropy_loss(
                self.global_projection, visual_embed, textual_embed, labels
            ),
            "global_align_loss": losses.global_align_loss(
                visual_embed, textual_embed, labels
            ),
            "local_align_loss": losses.global_align_loss_from_sim(
                local_similarity, labels
            ),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
