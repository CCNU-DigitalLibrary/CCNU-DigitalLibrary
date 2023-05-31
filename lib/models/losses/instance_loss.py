import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def cross_entropy_loss(projection, visual_embed, textual_embed, labels, scale=1, norm=False, epsilon=0.0,
                       projection_text=None, return_acc=False):
    if norm:
        visual_norm = F.normalize(visual_embed, p=2, dim=-1)
        textual_norm = F.normalize(textual_embed, p=2, dim=-1)
    else:
        visual_norm = visual_embed
        textual_norm = textual_embed
    projection_norm = F.normalize(projection, p=2, dim=0)
    if projection_text is not None:
        projection_text_norm = F.normalize(projection_text, p=2, dim=0)

    visual_logits = scale * torch.matmul(visual_norm, projection_norm)
    if projection_text is not None:
        textual_logits = scale * torch.matmul(textual_norm, projection_text_norm)
    else:
        textual_logits = scale * torch.matmul(textual_norm, projection_norm)

    if epsilon > 0:
        criterion = CrossEntropyLabelSmooth(num_classes=projection_norm.shape[1])
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(visual_logits, labels) + criterion(textual_logits, labels)

    if return_acc:
        with torch.no_grad():
            visual_acc = torch.argmax(visual_logits, dim=1).eq(labels).to(visual_logits.device).float().mean()
            textual_acc = torch.argmax(textual_logits, dim=1).eq(labels).to(textual_logits.device).float().mean()
        return loss, visual_acc, textual_acc
    return loss


def weighted_cross_entropy_loss(projection, visual_embed, textual_embed, label, weight):
    visual_embed = visual_embed @ projection
    textual_embed = textual_embed @ projection
    loss = F.cross_entropy(visual_embed, label, weight=weight) + F.cross_entropy(
        textual_embed, label, weight=weight
    )
    return loss
