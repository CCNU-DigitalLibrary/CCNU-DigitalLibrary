import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .loss import make_loss_evaluator
from lib.models.embeddings.basehead import BaseHead


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k=None, d_v=None, dropout=0.0):
        super().__init__()

        self.n_head = n_head
        if d_k is None: d_k = d_model
        if d_v is None: d_v = d_model
        self.d_k, self.d_v = d_k, d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, k_mask=None):
        q = F.normalize(q, p=2, dim=-1)  # b * n x lq x dv
        k = F.normalize(k, p=2, dim=-1)  # b * n x lk x dv
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if k_mask is not None:
            attn_output_weights.masked_fill(k_mask, float("-inf"))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if self.training:
            attn_output_weights = self.dropout(attn_output_weights)
        return torch.bmm(attn_output_weights, v), attn_output_weights  # b * n x lk x dv, b * n x lq x lk

    def forward_attention(self, q, k, v, k_mask=None):
        bs_q, bs_k, bs_v = q.size(0), k.size(0), v.size(0)
        len_q, len_k, len_v = q.size(-2), k.size(-2), v.size(-2)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b * n x lq x dv
        q = q.view(bs_q, len_q, self.n_head, self.d_k).transpose(1, 2).contiguous().view(bs_q * self.n_head, len_q,
                                                                                         self.d_k)
        k = k.view(bs_k, len_k, self.n_head, self.d_k).transpose(1, 2).contiguous().view(bs_k * self.n_head, len_k,
                                                                                         self.d_k)
        v = v.view(bs_v, len_v, self.n_head, self.d_v).transpose(1, 2).contiguous().view(bs_v * self.n_head, len_v,
                                                                                         self.d_v)

        if k_mask is not None:
            # b x lk
            k_mask = k_mask.view(bs_k, 1, len_k).repeat(1, self.n_head, 1).view(bs_k * self.n_head, 1, len_k)
        attn_output, attn_output_weights = self.scaled_dot_product_attention(q, k, v, k_mask)

        return attn_output.view(bs_q, self.n_head, len_v, self.d_v), attn_output_weights

    def forward(self, q, k, v, k_mask=None):
        q, k, v = self.w_qs(q), self.w_ks(k), self.w_vs(v)
        return self.forward_attention(q, k, v, k_mask)


class SAFAHead(BaseHead):
    """
    Implementation of `LEARNING SEMANTIC-ALIGNED FEATURE REPRESENTATION FOR TEXT-BASED PERSON SEARCH`
        - http://arxiv.org/abs/2112.06714
    """
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.num_heads = cfg.MODEL.SAFA.NUM_HEADS
        self.drop_rate = cfg.MODEL.SAFA.DROPOUT
        assert visual_size == textual_size == self.embed_size, \
            f'visual_size({visual_size}), textual_size({textual_size}), embed_size({self.embed_size}) must be same'

        self.shared_embed_layer = MultiHeadAttention(n_head=self.num_heads, d_model=self.embed_size,
                                                     dropout=self.drop_rate)

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def forward(self, visual_feature, textual_feature, captions, **kwargs):
        # visual_feature (B, S+1, C), textual_feature (B, L+1, C)
        # B x S x C
        visual_attn_output, visual_attn_output_weights = self.shared_embed_layer(visual_feature, visual_feature,
                                                                                 visual_feature)
        visual_embed = torch.chunk(visual_attn_output[:, :, 0], chunks=self.num_heads, dim=1)
        visual_embed = [embed.squeeze(dim=1) for embed in visual_embed]
        # B x L x C
        padding_mask = ~kwargs['text_mask'].to(torch.bool) if 'text_mask' in kwargs else None
        textual_attn_output, textual_attn_output_weights = self.shared_embed_layer(textual_feature, textual_feature,
                                                                 textual_feature,
                                                                 k_mask=padding_mask)
        textual_embed = torch.chunk(textual_attn_output[:, :, 0], chunks=self.num_heads, dim=1)
        textual_embed = [embed.squeeze(dim=1) for embed in textual_embed]

        visual_embed_global = visual_feature[:, 0]
        textual_embed_global = textual_feature[:, 0]

        if self.training:
            losses = self.loss_evaluator(
                visual_embed,
                textual_embed,
                captions,
                visual_embed_global,
                textual_embed_global,
            )
            return None, losses

        outputs = list()
        visual_embed.append(visual_embed_global)
        textual_embed.append(textual_embed_global)
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        # outputs.append(torch.stack(visual_embed, dim=1))
        # outputs.append(torch.stack(textual_embed, dim=1))
        return outputs, None


def build_safa_head(cfg, visual_size, textual_size):
    model = SAFAHead(cfg, visual_size, textual_size)
    return model
