import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses
from lib.models.losses import BaseLossComputation


class LossComputation(BaseLossComputation):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mixture = cfg.MODEL.LOSSES.GA.MIXTURE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK
        self.epsilon = cfg.MODEL.LOSSES.CE.EPSILON
        self.learn_scale = cfg.MODEL.LOSSES.GA.LEARN_SCALE
        self.loss_type = dict(zip(cfg.MODEL.EMBEDDING.LOSS_TYPE, cfg.MODEL.EMBEDDING.LOSS_WEIGHT))
        self.shared_projection = cfg.MODEL.LOSSES.CE.SHARED_PROJECTION
        self.embed_style = cfg.MODEL.INVERT.EMBEDDING.STYLE
        self.inver_style = cfg.MODEL.INVERT.INVERT_NET.STYLE
        self.learn_temp = cfg.MODEL.LOSSES.CONTRAST.LEARNABLE_TEMP
        self.learn_dsl_temp = cfg.MODEL.LOSSES.DSL.LEARNABLE_TEMP

        if self.learn_temp:
            self.contrast_temp = Parameter(torch.tensor(0.05).log(), requires_grad=True)
        else:
            self.contrast_temp = torch.tensor(0.05).log()

        if self.learn_dsl_temp:
            self.dual_softmax_temp = Parameter(torch.tensor(1000).log(), requires_grad=True)
        else:
            self.dual_softmax_temp = torch.tensor(1000).log()

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
        if not self.shared_projection:
            self.projection_text = Parameter(
                torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
                requires_grad=True,
            )
            nn.init.xavier_uniform_(self.projection_text.data, gain=1)
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        v2t_inter,
        t2v_inter,
        captions,
        visual_embeds_bn=None,
        textual_embeds_bn=None,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {}
        # re-order the states
        if self.embed_style in ['multi', 'mha']:
            if self.inver_style == 'transformer':
                v2t_inter = [[t[:, i] for i in range(t.size(1))] for t in v2t_inter]
                t2v_inter = [[v[:, i] for i in range(v.size(1))] for v in t2v_inter]
            v2t_inter = sum(v2t_inter, [])
            t2v_inter_reverse = sum([list(reversed(t2v_inter_i)) for t2v_inter_i in t2v_inter], [])
        else:
            t2v_inter_reverse = list(reversed(t2v_inter))

        if "cos_dist_loss" in self.loss_type:
            cos_dist_loss = 0
            for v2t, t2v in zip(v2t_inter, t2v_inter_reverse):
                cos_dist_loss += losses.cosine_dist(v2t, t2v.detach()).mean() + \
                            losses.cosine_dist(v2t.detach(), t2v).mean()
            loss.update({"cos_dist_loss": self.loss_type["cos_dist_loss"] * cos_dist_loss})

        if "contrastive_loss" in self.loss_type:
            contrastive_loss = 0
            for v2t, t2v in zip(v2t_inter, t2v_inter_reverse):
                contrastive_loss += losses.contrastive_loss(v2t, t2v.detach(), labels, temp=self.contrast_temp) + \
                                    losses.contrastive_loss(v2t.detach(), t2v, labels, temp=self.contrast_temp)
            loss.update({"contrastive_loss": self.loss_type["contrastive_loss"] * contrastive_loss})

        if "dual_softmax_loss" in self.loss_type:
            dual_softmax_loss = 0
            for v2t, t2v in zip(v2t_inter, t2v_inter_reverse):
                dual_softmax_loss += losses.dual_softmax_loss(v2t, t2v, temp=self.dual_softmax_temp)
            loss.update({"dual_softmax_loss": self.loss_type["dual_softmax_loss"] * dual_softmax_loss})

        if "merged_contrastive_loss" in self.loss_type:
            merged_contrastive_loss = 0
            for v2t, t2v in zip(v2t_inter, t2v_inter_reverse):
                merged_contrastive_loss += losses.contrastive_loss(v2t, t2v, labels, merged=True, temp=self.contrast_temp)
            loss.update({"merged_contrastive_loss": self.loss_type["merged_contrastive_loss"] * merged_contrastive_loss})

        # intra modal losses
        if "instance_loss" in self.loss_type and self.bnneck:
            instance_loss = 0
            visual_acc = []
            textual_acc = []
            if not(isinstance(visual_embeds_bn, list) and isinstance(textual_embeds_bn, list)):
                visual_embeds_bn = [visual_embeds_bn]
                textual_embeds_bn = [textual_embeds_bn]
            for visual_embed_bn, textual_embed_bn in zip(visual_embeds_bn, textual_embeds_bn):
                cur_instance_loss, cur_visual_acc, cur_textual_acc = losses.cross_entropy_loss(
                    self.projection,
                    visual_embed_bn,
                    textual_embed_bn,
                    labels,
                    epsilon=self.epsilon,
                    projection_text=self.projection_text,
                    return_acc=True
                )
                instance_loss += cur_instance_loss
                visual_acc.append(cur_visual_acc)
                textual_acc.append(cur_textual_acc)
            loss.update({"instance_loss": self.loss_type["instance_loss"] * instance_loss,
                         "visual_acc": sum(visual_acc) / len(visual_acc),
                         "textual_acc": sum(textual_acc) / len(textual_acc)})

        if "triplet_loss" in self.loss_type:
            triplet_loss = 0
            triplet = losses.TripletLoss(self.cfg)
            for v2t, t2v in zip(v2t_inter, t2v_inter_reverse):
                triplet_loss += triplet(v2t, labels) + triplet(t2v, labels)
            loss.update({"triplet_loss": self.loss_type["triplet_loss"] * triplet_loss})
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
