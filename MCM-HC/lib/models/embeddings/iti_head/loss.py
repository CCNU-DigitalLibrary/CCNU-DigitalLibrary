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
        self.shared_projection = cfg.MODEL.LOSSES.CE.SHARED_PROJECTION

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

        self.projection_text = None
        if self.shared_projection:
            self.projection_text = Parameter(
                torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
                requires_grad=True,
            )
            nn.init.xavier_uniform_(self.projection_text.data, gain=1)
        nn.init.xavier_uniform_(self.projection.data, gain=1)

        self.anneal = cfg.MODEL.ITI.ANNEAL
        self.num_epochs = cfg.SOLVER.NUM_EPOCHS - cfg.SOLVER.WARMUP_EPOCHS
        self.cfg = cfg

    def forward(
            self,
            visual_embed,
            textual_embed,
            captions,
            epoch,
            visual_embed_bn=None,
            textual_embed_bn=None,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {}
        if self.anneal:
            cross_anneal_factor = epoch / self.num_epochs
            intra_anneal_factor = 1 - cross_anneal_factor
        else:
            intra_anneal_factor = cross_anneal_factor = 1

        # intra modal losses
        if "instance_loss" in self.loss_type and self.bnneck:
            loss.update({"instance_loss": intra_anneal_factor * self.loss_type["instance_loss"] * losses.cross_entropy_loss(
                self.projection,
                visual_embed_bn,
                textual_embed_bn,
                labels,
                epsilon=self.epsilon,
                projection_text=self.projection_text
            )})

        if "triplet_loss" in self.loss_type:
            loss.update({"triplet_loss": intra_anneal_factor * self.loss_type["triplet_loss"] *
                                         (losses.TripletLoss(self.cfg)(visual_embed, labels) +
                                          losses.TripletLoss(self.cfg)(textual_embed, labels))
                         })

        if "softmax_triplet_loss" in self.loss_type:
            loss.update({"softmax_triplet_loss": intra_anneal_factor * self.loss_type["softmax_triplet_loss"] *
                                                 (losses.SoftmaxTripletLoss(self.cfg)(visual_embed, labels) +
                                                  losses.SoftmaxTripletLoss(self.cfg)(textual_embed, labels))
                         })

        if "softsoftmax_triplet_loss" in self.loss_type:
            loss.update({"softsoftmax_triplet_loss": intra_anneal_factor * self.loss_type["softsoftmax_triplet_loss"] *
                                                     (losses.SoftSoftmaxTripletLoss(self.cfg)(visual_embed, labels) +
                                                      losses.SoftSoftmaxTripletLoss(self.cfg)(textual_embed, labels))
                         })

        # cross modal losses
        if "global_align_loss" in self.loss_type and self.bnneck:
            loss.update({"global_align_loss": cross_anneal_factor * self.loss_type[
                "global_align_loss"] * losses.global_align_loss(
                visual_embed_bn, textual_embed_bn, labels, self.mixture,
                scale_pos=self.scale_pos,
                scale_neg=self.scale_neg,
            )})

        if "cmpm_loss" in self.loss_type:
            loss.update({"cmpm_loss": cross_anneal_factor * self.loss_type["cmpm_loss"] * losses.cmpm_loss(visual_embed,
                                                                                                           textual_embed,
                                                                                                           labels)})

        if "cmpc_loss" in self.loss_type:
            loss.update({"cmpc_loss": cross_anneal_factor * self.loss_type["cmpc_loss"] * losses.cmpc_loss(
                self.projection, visual_embed, textual_embed, labels)})

        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
