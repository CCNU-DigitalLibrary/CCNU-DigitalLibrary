import torch
import torch.nn.functional as F
import numpy as np


def dual_softmax_loss(visual_embed, textual_embed, temp, sim_matrix=None, verbose=False):
    """
    dual softmax Loss
    http://arxiv.org/abs/2109.04290
    :param visual_embed:
    :param textual_embed:
    :param temp: temperature used in softmax
    :param sim_matrix: similarity matrix, if None, calculated by visual and textual embeddings, else process it directly.
    :param verbose: return the accuracy.
    :param epsilon
    :return
        loss: dual softmax loss
        accuracy(optional): top-1 acc.
    """
    if sim_matrix:
        # With an appropriate temperature parameter, the models achieves higher performance
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp.exp(), dim=0) * len(sim_matrix)
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
    else:
        temp = temp.exp()
        visual_embed = visual_embed.float()
        textual_embed = textual_embed.float()
        visual_embed_norm = F.normalize(visual_embed, p=2, dim=-1)
        textual_embed_norm = F.normalize(textual_embed, p=2, dim=-1)
        sim_matrix = torch.matmul(visual_embed_norm, textual_embed_norm.t())

        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp.exp(), dim=0) * len(sim_matrix)
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt

    return loss