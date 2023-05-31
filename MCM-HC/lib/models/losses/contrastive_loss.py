import torch
import torch.nn.functional as F
import numpy as np


def contrastive_loss(embeds1, embeds2, labels, temp, verbose=False, epsilon=1e-8, merged=False):
    """
    Contrastive Loss
    :param embeds1:
    :param embed2:
    :param labels
    :return
        loss: contrastive loss
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """
    temp = temp.exp()
    embeds1 = embeds1.float()
    embeds2 = embeds2.float()
    batch_size = embeds1.shape[0]

    labels_reshape = torch.reshape(labels, (batch_size, 1))

    if merged:
        # concat embeds1 and embeds2
        labels_reshape = torch.cat([labels_reshape, labels_reshape])
        embeds1 = torch.cat([embeds1, embeds2])
        embeds2 = embeds1.clone()
        # remove the self similarity
        self_mask = torch.eye(batch_size * 2).to(labels_reshape.device).to(torch.bool)

    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = labels_dist == 0

    embeds1_norm = F.normalize(embeds1, p=2, dim=-1)
    embeds2_norm = F.normalize(embeds2, p=2, dim=-1)
    embeds_simi = torch.matmul(embeds1_norm, embeds2_norm.t())

    if merged:
        labels_mask = labels_mask[~self_mask].reshape(labels_mask.size(0), -1)
        embeds_simi = embeds_simi[~self_mask].reshape(embeds_simi.size(0), -1)

    # normalize the true matching distribution
    labels_mask_norm = F.normalize(labels_mask.float(), p=1, dim=1)

    embeds_simi_softmax = F.softmax(embeds_simi / temp, dim=1)
    contrastive_loss = embeds_simi_softmax * (
            F.log_softmax(embeds_simi / temp, dim=1) - torch.log(labels_mask_norm + epsilon))
    loss = contrastive_loss.sum(dim=1).mean()

    if not np.isfinite(loss.detach().cpu().numpy()):
        print(contrastive_loss, loss)
        raise FloatingPointError
    if verbose:
        # cosine similarity
        pos_avg_sim = torch.mean(torch.masked_select(embeds_simi, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(embeds_simi, labels_mask == 0))

        return loss, pos_avg_sim, neg_avg_sim
    return loss