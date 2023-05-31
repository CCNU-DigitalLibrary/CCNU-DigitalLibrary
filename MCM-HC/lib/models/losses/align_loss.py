import torch
import torch.nn.functional as F


def global_align_loss(
        visual_embed,
        textual_embed,
        labels,
        mixture=False,
        alpha=0.6,
        beta=0.4,
        scale_pos=10,
        scale_neg=40,
):
    batch_size = labels.size(0)
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    similarity = torch.matmul(visual_norm, textual_norm.t())
    labels_ = (
        labels.expand(batch_size, batch_size)
            .eq(labels.expand(batch_size, batch_size).t())
            .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    if mixture:
        margin = alpha - beta
        tmp = similarity
        tmp[neg_inds] = 1
        hard_v_pos, _ = torch.min(tmp, dim=1)
        hard_t_pos, _ = torch.min(tmp, dim=0)
        tmp = similarity
        tmp[pos_inds] = 0
        hard_v_neg, _ = torch.max(tmp, dim=1)
        hard_t_neg, _ = torch.max(tmp, dim=0)
        #         y = torch.ones_like(hard_v_neg)
        #         loss_v_dist = F.margin_ranking_loss(hard_v_neg, hard_v_pos, y, margin=margin, reduction="sum")
        #         loss_t_dist = F.margin_ranking_loss(hard_t_neg, hard_t_pos, y, margin=margin, reduction="sum")
        v_dist = hard_v_pos - hard_v_neg
        t_dist = hard_t_pos - hard_t_neg
        loss_v_dist = torch.log(1 + torch.exp(margin - v_dist))
        loss_t_dist = torch.log(1 + torch.exp(margin - t_dist))
        loss = loss + loss_t_dist.sum() + loss_v_dist.sum()

    loss /= batch_size
    return loss


def global_align_loss_from_sim(
        similarity,
        labels,
        alpha=0.6,
        beta=0.4,
        scale_pos=10,
        scale_neg=40,
):
    batch_size = labels.size(0)
    labels_ = (
        labels.expand(batch_size, batch_size)
            .eq(labels.expand(batch_size, batch_size).t())
            .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    loss /= batch_size
    return loss


def local_align_no_sampling_loss(
        part_embed,
        attr_embed,
        labels,
        part_masks,
        attr_masks,
        num_parts=5,
        alpha=0.6,
        beta=0.4,
        scale_pos=10,
        scale_neg=40,
):
    batch_size = labels.size(0)
    part_embed = F.normalize(part_embed, p=2, dim=2)
    attr_embed = F.normalize(attr_embed, p=2, dim=2)
    labels_ = labels.expand(batch_size, batch_size).eq(
        labels.expand(batch_size, batch_size).t()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0

    local_loss = 0.0
    for i in range(num_parts):
        filter_inds = torch.ones_like(labels_)
        filter_inds[~attr_masks[:, i], :] = 0
        filter_inds[:, ~part_masks[:, i]] = 0
        filter_pos_inds = filter_inds & pos_inds
        filter_neg_inds = filter_inds & neg_inds

        local_similarity = torch.matmul(attr_embed[i], part_embed[i].t())
        loss_pos = torch.log(
            1 + torch.exp(-scale_pos * (local_similarity[filter_pos_inds] - alpha))
        )
        loss_neg = torch.log(
            1 + torch.exp(scale_neg * (local_similarity[filter_neg_inds] - beta))
        )
        local_loss += (loss_pos.sum() + loss_neg.sum()) * 2.0
    return local_loss / batch_size / num_parts


def local_align_loss(
        part_embed,
        attribute_embed,
        labels,
        part_masks,
        attr_masks,
        num_parts=5,
        alpha=0.6,
        beta=0.4,
        scale_pos=10,
        scale_neg=40,
        topK=8,
):
    batch_size = labels.size(0)
    part_embed = F.normalize(part_embed, p=2, dim=2)
    attribute_embed = F.normalize(attribute_embed, p=2, dim=2)
    labels_ = labels.expand(batch_size, batch_size).eq(
        labels.expand(batch_size, batch_size).t()
    )

    losses = 0
    for i in range(num_parts):
        part_mask = part_masks[:, i]
        attr_mask = attr_masks[:, i]
        similarity = torch.matmul(part_embed[i], attribute_embed[i].t())
        rank1 = torch.argsort(similarity, dim=1, descending=True)
        rank2 = torch.argsort(similarity.t(), dim=1, descending=True)

        loss = 0
        for j in range(batch_size):
            if part_mask[j] == 0:
                continue
            pred = similarity[j, attr_mask]
            # k-reciprocal sample
            label = labels_[j, :].float()
            forward_k_idx = rank1[i, :topK]
            backward_k_idx = rank2[forward_k_idx, :topK]
            sample_pos_idx = torch.nonzero(backward_k_idx == i)[:, 0]
            sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
            label[sample_pos_idx] = 1
            label = label[attr_mask]
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            if pos_inds.numel() > 0:
                loss_pos = torch.log(
                    1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha))
                )
                loss += loss_pos.sum()
            if neg_inds.numel() > 0:
                loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                loss += loss_neg.sum()

            if attr_mask[j] == 0:
                continue
            pred = similarity[part_mask, j]
            # k-reciprocal sample
            label = labels_[j, :].float()
            forward_k_idx = rank2[i, :topK]
            backward_k_idx = rank1[forward_k_idx, :topK]
            sample_pos_idx = torch.nonzero(backward_k_idx == i)[:, 0]
            sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
            label[sample_pos_idx] = 1
            label = label[part_mask]
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            if pos_inds.numel() > 0:
                loss_pos = torch.log(
                    1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha))
                )
                loss += loss_pos.sum()
            if neg_inds.numel() > 0:
                loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                loss += loss_neg.sum()

        loss /= batch_size
        losses += loss
    losses /= num_parts
    return losses
