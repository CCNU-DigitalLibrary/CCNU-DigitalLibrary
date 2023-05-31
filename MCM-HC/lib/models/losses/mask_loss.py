import torch
import torch.nn.functional as F


def ce_mask_loss(seg_feat, masks, num_parts=5):
    masks = torch.stack(masks, dim=1)  # 5 * b * h/8 * w/8
    masks = masks.view(-1, masks.size(-2), masks.size(-1))  # 5b * h/8 * w/8

    loss = F.cross_entropy(seg_feat, masks.long(), reduction="none")
    loss = num_parts * loss.mean()
    return loss


def bce_mask_loss(seg_feat, masks, num_parts=5):
    loss = F.binary_cross_entropy_with_logits(seg_feat, masks, reduction="mean")
    return loss * num_parts
