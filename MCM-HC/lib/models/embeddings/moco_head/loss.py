import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import lib.models.losses as losses


def infonce_loss(
    v_pos,
    v_neg,
    t_pos,
    t_neg,
    T=0.07,
):
    v_logits = torch.cat([v_pos, v_neg], dim=1) / T
    t_logits = torch.cat([t_pos, t_neg], dim=1) / T
    labels = torch.zeros(v_logits.shape[0], dtype=torch.long).cuda()
    loss = F.cross_entropy(v_logits, labels) + F.cross_entropy(t_logits, labels)
    return loss


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        self.epsilon = cfg.MODEL.LOSSES.CE.EPSILON
        # self.T = Parameter(torch.tensor(0.07), requires_grad=True)
        self.T = 0.07
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(self, v_embed, t_embed, v_pos, v_neg, t_pos, t_neg, labels):
        loss = {
            "instance_loss": losses.cross_entropy_loss(
                self.projection,
                v_embed,
                t_embed,
                labels,
                epsilon=self.epsilon,
            ),
            "infonce_loss": infonce_loss(
                v_pos,
                v_neg,
                t_pos,
                t_neg,
                self.T,
            ),
            "global_align_loss": losses.global_align_loss(v_embed, t_embed, labels),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
