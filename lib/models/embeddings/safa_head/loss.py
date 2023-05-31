import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.loss_type = dict(zip(cfg.MODEL.EMBEDDING.LOSS_TYPE, cfg.MODEL.EMBEDDING.LOSS_WEIGHT))

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
        visual_embed_global=None,
        textual_embed_global=None,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {}
        if "cmpm_loss" in self.loss_type:
            cmpm_loss = []
            for visual_feat, textual_feat in zip(visual_embed, textual_embed):
                cmpm_loss.append(losses.cmpm_loss(visual_feat, textual_feat, labels))
            cmpm_loss.append(losses.cmpm_loss(visual_embed_global, textual_embed_global, labels))
            loss.update({"cmpm_loss": self.loss_type["cmpm_loss"] * sum(cmpm_loss)})

        if "cmpc_loss" in self.loss_type:
            cmpc_loss = []
            for visual_feat, textual_feat in zip(visual_embed, textual_embed):
                cmpc_loss.append(losses.cmpc_loss(
                    self.projection, visual_feat, textual_feat, labels)
                    )
            cmpc_loss.append(losses.cmpc_loss(self.projection, visual_embed_global, textual_embed_global, labels))
            loss.update({"cmpc_loss": self.loss_type["cmpc_loss"] * sum(cmpc_loss)})

        if "diversity_reg_loss" in self.loss_type:
            loss.update({"diversity_reg_loss": self.loss_type["diversity_reg_loss"] * losses.diversity_reg(visual_embed,
                                                                                                           textual_embed)})

        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
