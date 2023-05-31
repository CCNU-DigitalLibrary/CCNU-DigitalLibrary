import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_parts = cfg.MODEL.NUM_PARTS

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        part_embed,
        attribute_embed,
        seg_feat,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        masks = [caption.get_field("crops") for caption in captions]
        vmask = torch.stack([caption.get_field("mask") for caption in captions])
        attributes = [caption.get_field("attribute") for caption in captions]
        tmask = torch.stack([attribute.get_field("mask") for attribute in attributes])

        loss = {
            "instance_loss": losses.cross_entropy_loss(
                self.projection, visual_embed, textual_embed, labels
            ),
            "mask_loss": losses.ce_mask_loss(seg_feat, masks, self.num_parts),
            "global_align_loss": losses.global_align_loss(
                visual_embed, textual_embed, labels
            ),
            "local_align_loss": losses.local_align_loss(
                part_embed, attribute_embed, labels, vmask, tmask, self.num_parts
            ),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
