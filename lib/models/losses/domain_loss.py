import torch
import torch.nn as nn


def domain_loss(visual_domain_logits, textual_domain_logits):
    criterion = nn.CrossEntropyLoss()
    batch_size = visual_domain_logits.shape[0]
    visual_domain_labels = torch.zeros(batch_size).long().cuda()
    textual_domain_labels = torch.ones(batch_size).long().cuda()
    loss = criterion(visual_domain_logits, visual_domain_labels) + criterion(
        textual_domain_logits, textual_domain_labels
    )
    return loss
