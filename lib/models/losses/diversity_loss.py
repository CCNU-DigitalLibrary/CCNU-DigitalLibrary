import torch
import torch.nn.functional as F


def diversity_reg(visual_embeds, textual_embeds):
    """
    L_div = \frac{1}{K\times (K-1)} \sum_{i=1}^{K} \sum_{j=1,i \neq j}^{K} \frac{e_i \cdot e_j}{||e_i||_2 ||e_j||_2} + \frac{t_i \cdot t_j}{||t_i||_2 ||t_j||_2}
    """
    if isinstance(visual_embeds, torch.Tensor):
        assert isinstance(textual_embeds, torch.Tensor)
        visual_embeds = list(visual_embeds)
        textual_embeds = list(textual_embeds)

    visual_embeds = torch.stack(visual_embeds, dim=1)
    textual_embeds = torch.stack(textual_embeds, dim=1)
    assert visual_embeds.size() == textual_embeds.size(), "visual_size({}) != textual_size({})".format(
        visual_embeds.size(), textual_embeds.size())
    visual_embeds_norm = F.normalize(visual_embeds, p=2, dim=-1)
    textual_embeds_norm = F.normalize(textual_embeds, p=2, dim=-1)
    visual_embeds_simi = torch.bmm(visual_embeds_norm, visual_embeds_norm.transpose(1, 2))
    textual_embeds_simi = torch.bmm(textual_embeds_norm, textual_embeds_norm.transpose(1, 2))
    # diagonal mask
    diag = 1 - torch.eye(visual_embeds_simi.size(1)).to(visual_embeds).expand_as(visual_embeds_simi)
    loss = (visual_embeds_simi + textual_embeds_simi) * diag / diag.sum()
    return loss.sum()
