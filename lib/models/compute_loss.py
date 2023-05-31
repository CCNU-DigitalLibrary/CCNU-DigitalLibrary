import torch
from torch.nn import functional as F
import torch.nn as nn


class MaskLoss:
    def __init__(self):
        self.mim_loss_function = F.l1_loss

        self.mlm_loss_function = nn.CrossEntropyLoss()

    def compute_mim_loss(self, origin_image=None, decoder_output=None, mask=None):
        loss_mim = F.mse_loss(origin_image, decoder_output)
        # loss_recon = self.mim_loss_function(origin_image, decoder_output, reduction="none")
        # loss_mim = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3
        return loss_mim
    def compute_mlm_loss(self,prediction_score=None, label=None):
        loss_mlm = self.mlm_loss_function(prediction_score, label)
        return loss_mlm


class HashLoss:

    def __init__(self):
        pass
    def compute_hash_loss(self, visual_embed, textual_embed, device=torch.device("cuda")):
        t_ones = torch.ones(visual_embed.shape).to(device, dtype=torch.float)

        fro_n = torch.norm(visual_embed * t_ones + textual_embed * t_ones, p=2, dim=1)
        balance_loss = torch.sum(fro_n)

        Bx = 2*(visual_embed >= 0).to(device, dtype=torch.float)-1
        By = 2*(textual_embed >= 0).to(device, dtype=torch.float)-1
        hash_loss = nn.MSELoss()(Bx, visual_embed) + nn.MSELoss()(By, textual_embed)
        # hash_loss = nn.MSELoss()(Bx, visual_embed)
        # hash_loss = nn.MSELoss()(By, textual_embed)
        return balance_loss, hash_loss


    def compute_similarity_loss(self, f1, f2, sim = None, alpha=1,device=torch.device("cuda")):
        batch_size, f_dim = f1.shape
        sim = torch.ones((batch_size, 1)).to(device, dtype=torch.float)
        inner = torch.bmm(f1.view(batch_size, 1, f_dim), f2.view(batch_size, f_dim, 1))
        # inner = torch.clamp(torch.bmm(f1.view(batch_size, 1, f_dim), f2.view(batch_size, f_dim, 1)),-1.5e1, 1.5e1)
        t_ones = torch.ones(batch_size, 1).to(device, dtype=torch.float)
        similarity_loss = torch.mean(torch.log(torch.add(t_ones, torch.exp(inner))) - alpha * torch.mul(sim, inner))
        return similarity_loss

    def compute_quant_loss(self, visual_output_embed, textual_output_embed):
        B_code = (visual_output_embed.detach() + textual_output_embed.detach()) / 2
        B_code = B_code.sign()
        quant_loss = torch.sum(torch.pow(B_code - visual_output_embed, 2)) + torch.sum(torch.pow(B_code - textual_output_embed, 2))
        return quant_loss






