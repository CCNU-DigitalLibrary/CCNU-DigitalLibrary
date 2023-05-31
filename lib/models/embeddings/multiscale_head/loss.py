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
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embeds,
        textual_embeds,
        captions,
        visual_embed_bn=None,
        textual_embed_bn=None,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {}
        if "cmpm_loss" in self.loss_type:
            cmpm_loss = []
            for visual_feat, textual_feat in zip(visual_embeds, textual_embeds):
                cmpm_loss.append(losses.cmpm_loss(visual_feat, textual_feat, labels))
            loss.update({"cmpm_loss": self.loss_type["cmpm_loss"] * sum(cmpm_loss)})

        if self.bnneck:
            visual_feat = visual_embed_bn
            textual_feat = textual_embed_bn
        else:
            visual_feat = visual_embeds[-1]
            textual_feat = textual_embeds[-1]
        if "instance_loss" in self.loss_type:
            instance_loss = []
            instance_loss.append(losses.cross_entropy_loss(
                self.projection,
                visual_feat,
                textual_feat,
                labels,
                epsilon=self.epsilon,
            ))
            loss.update({"instance_loss": self.loss_type["instance_loss"] * sum(instance_loss)})

        if "global_align_loss" in self.loss_type:
            global_align_loss = []
            global_align_loss.append(losses.global_align_loss(
                    visual_feat, textual_feat, labels, self.mixture
                ))
            loss.update({"global_align_loss": self.loss_type["global_align_loss"] * sum(global_align_loss)})

        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
