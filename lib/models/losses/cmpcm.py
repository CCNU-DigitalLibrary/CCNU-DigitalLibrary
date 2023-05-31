import torch
import torch.nn as nn
import torch.nn.functional as F


def cmpc_loss(projection, visual_embed, textual_embed, labels, verbose=False):
    """
    Cross-Modal Projection Classfication loss (CMPC)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
    """
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    projection_norm = F.normalize(projection, p=2, dim=0)

    image_proj_text = (
            torch.sum(visual_embed * textual_norm, dim=1, keepdim=True) * textual_norm
    )
    text_proj_image = (
            torch.sum(textual_embed * visual_norm, dim=1, keepdim=True) * visual_norm
    )

    image_logits = torch.matmul(image_proj_text, projection_norm)
    text_logits = torch.matmul(text_proj_image, projection_norm)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    # classification accuracy for observation
    if verbose:
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return loss, image_precision, text_precision
    return loss


def cmpm_loss(visual_embed, textual_embed, labels, verbose=False, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = visual_embed.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = labels_dist == 0

    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    image_proj_text = torch.matmul(visual_embed, textual_norm.t())
    text_proj_image = torch.matmul(textual_embed, visual_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (
            F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (
            F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
        torch.sum(t2i_loss, dim=1)
    )

    if verbose:
        sim_cos = torch.matmul(visual_norm, textual_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return loss, pos_avg_sim, neg_avg_sim
    return loss
