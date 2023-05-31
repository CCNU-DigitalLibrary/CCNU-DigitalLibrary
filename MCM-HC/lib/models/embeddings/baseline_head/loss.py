import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mixture = cfg.MODEL.LOSSES.GA.MIXTURE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK
        self.epsilon = cfg.MODEL.LOSSES.CE.EPSILON
        self.learn_scale = cfg.MODEL.LOSSES.GA.LEARN_SCALE
        self.loss_type = dict(zip(cfg.MODEL.EMBEDDING.LOSS_TYPE, cfg.MODEL.EMBEDDING.LOSS_WEIGHT))

        if self.learn_scale:
            self.scale_pos = Parameter(torch.tensor(10.0), requires_grad=True)
            self.scale_neg = Parameter(torch.tensor(40.0), requires_grad=True)
        else:
            self.scale_pos = 10.0
            self.scale_neg = 40.0

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            # torch.randn(256, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
            self,
            visual_embed,
            textual_embed,
            captions,
            visual_embed_bn=None,
            textual_embed_bn=None,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {}
        if "cmpm_loss" in self.loss_type:
            loss.update(
                {"cmpm_loss": self.loss_type["cmpm_loss"] * losses.cmpm_loss(visual_embed, textual_embed, labels)})

        if "cmpc_loss" in self.loss_type:
            loss.update({"cmpc_loss": self.loss_type["cmpc_loss"] * losses.cmpc_loss(self.projection, visual_embed,
                                                                                     textual_embed, labels)})

        if "instance_loss" in self.loss_type and self.bnneck:
            loss.update({"instance_loss": self.loss_type["instance_loss"] * losses.cross_entropy_loss(
                self.projection,
                visual_embed_bn,
                textual_embed_bn,
                labels,
                epsilon=self.epsilon,
            )})

        if "global_align_loss" in self.loss_type and self.bnneck:
            loss.update({"global_align_loss": self.loss_type["global_align_loss"] * losses.global_align_loss(
                visual_embed_bn, textual_embed_bn, labels, self.mixture
            )})

        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
