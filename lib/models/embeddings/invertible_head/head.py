import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.models.embeddings.basehead import BaseHead, conv1x1, Bottleneck
from lib.models.layers.flow import RevSymmCouplingFlow, MSRevSymmCouplingFlow, RevTransformerFlow
from .loss import make_loss_evaluator


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

    def forward_attention(self, q, k, v, k_mask=None, need_weights=True):
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
        attn_output = attn_output.view(bs_q, self.n_head, len_q, self.d_v)
        if need_weights:
            attn_output_weights = attn_output_weights.view(bs_q, self.n_head, len_q, len_k)
            attn_output_weights = attn_output_weights.sum(dim=1) / self.n_head
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    def forward(self, q, k, v, k_mask=None):
        q, k, v = self.w_qs(q), self.w_ks(k), self.w_vs(v)
        return self.forward_attention(q, k, v, k_mask)


class InvertHead(BaseHead):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK
        self.text_mask = cfg.MODEL.EMBEDDING.TEXT_MASK
        self.embed_style = cfg.MODEL.INVERT.EMBEDDING.STYLE
        self.invert_style = cfg.MODEL.INVERT.INVERT_NET.STYLE
        self.output_all_states = cfg.MODEL.INVERT.TEST.OUTPUT_ALL_STATES
        self.output_all_scales = cfg.MODEL.INVERT.TEST.OUTPUT_ALL_SCALES
        self.drop_rate = cfg.MODEL.INVERT.EMBEDDING.DROPOUT
        self.bnneck_global = cfg.MODEL.INVERT.EMBEDDING.BOTTLENECK_GLOBAL

        if self.embed_style == 'single':
            self.visual_embed_layer = nn.Sequential(
                nn.Linear(visual_size, self.embed_size),
                nn.BatchNorm1d(self.embed_size))
            self.textual_embed_layer = nn.Sequential(
                nn.Linear(textual_size, self.embed_size),
                nn.BatchNorm1d(self.embed_size))
        elif self.embed_style == 'multi':
            self.branch_size = cfg.MODEL.INVERT.EMBEDDING.NUM_BRANCHES
            # visual_layers
            if self.embed_size == 1024:
                self.visual_embed_layer_lower = nn.Identity()
            else:
                self.visual_embed_layer_lower = nn.Sequential(
                    conv1x1(1024, self.embed_size),
                    nn.BatchNorm2d(self.embed_size),
                    nn.ReLU(inplace=True))
            self.visual_embed_layer = nn.Sequential(
                conv1x1(visual_size, self.embed_size),
                nn.BatchNorm2d(self.embed_size),
                nn.ReLU(inplace=True))
            # textual_layers
            self.textual_embed_layer = nn.Sequential(
                conv1x1(textual_size, self.embed_size),
                nn.BatchNorm2d(self.embed_size),
                nn.ReLU(inplace=True))
            self.text_branches_type = cfg.MODEL.INVERT.EMBEDDING.TEXT_BRANCHES_TYPE
            self.build_text_branch()
        elif self.embed_style == 'mha':
            self.num_heads = cfg.MODEL.INVERT.EMBEDDING.NUM_BRANCHES
            self.visual_embed_layer = MultiHeadAttention(n_head=self.num_heads, d_model=self.embed_size,
                                                         dropout=self.drop_rate)
            self.textual_embed_layer = MultiHeadAttention(n_head=self.num_heads, d_model=self.embed_size,
                                                         dropout=self.drop_rate)
        else:
            raise NotImplementedError(f'Unsupported embedding style {self.embed_style}')

        # used in NAFS, test whether it is necessary
        if self.bnneck_global:
            self.bottleneck_image = nn.BatchNorm1d(self.embed_size)
            self.bottleneck_image.bias.requires_grad_(False)
            self.bottleneck_text = nn.BatchNorm1d(self.embed_size)
            self.bottleneck_text.bias.requires_grad_(False)

        visual_pool_type = cfg.MODEL.EMBEDDING.VISUAL_POOL
        textual_pool_type = cfg.MODEL.EMBEDDING.TEXTUAL_POOL
        self.construct_pool_layer(visual_pool_type, textual_pool_type)

        if self.bnneck:
            self.visual_bnneck = nn.BatchNorm1d(self.embed_size)
            self.textual_bnneck = nn.BatchNorm1d(self.embed_size)
            self.visual_bnneck.bias.requires_grad_(False)  # no shift
            self.textual_bnneck.bias.requires_grad_(False)  # no shift

        self.invertible_net = self.build_invertible_network(cfg)
        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def build_text_branch(self):
        self.text_branches = nn.ModuleList()
        if self.text_branches_type == 'conv':
            for _ in range(self.branch_size):
                downsample = nn.Sequential(
                    conv1x1(self.embed_size, self.embed_size),
                    nn.BatchNorm2d(self.embed_size))
                self.text_branches.append(
                    nn.Sequential(
                        Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=512, downsample=downsample),
                        Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=512),
                        Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=512)))
        elif self.text_branches_type == 'transformer':
            self.text_branches = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=1)] * self.branch_size
            )
        else:
            raise NotImplementedError(f'{self.text_branches_type} is not supported.')

    def build_invertible_network(self, cfg):
        if cfg.MODEL.INVERT.INVERT_NET.STYLE == 'coupling':
            if cfg.MODEL.INVERT.EMBEDDING.STYLE == 'single':
                return RevSymmCouplingFlow(cfg)
            elif cfg.MODEL.INVERT.EMBEDDING.STYLE == 'multi':
                return MSRevSymmCouplingFlow(cfg)
            elif cfg.MODEL.INVERT.EMBEDDING.STYLE == 'mha':
                return MSRevSymmCouplingFlow(cfg, num_ends=cfg.MODEL.INVERT.EMBEDDING.NUM_BRANCHES + 1)
            else:
                raise NotImplementedError(f'{cfg.MODEL.INVERT.EMBEDDING.STYLE} is not supported.')
        elif cfg.MODEL.INVERT.INVERT_NET.STYLE == 'transformer':
            assert cfg.MODEL.INVERT.EMBEDDING.STYLE in ['multi', 'mha'], \
                'multiple features should be extracted when use transformer in invertible neural network'
            return RevTransformerFlow(cfg)
        else:
            raise NotImplementedError(f'{cfg.MODEL.INVERT.INVERT_NET.STYLE} is not supported')

    def forward(self, visual_feature, textual_feature, captions, **kwargs):
        # extract embeddings
        if self.embed_style == 'single':
            batch_size = visual_feature.size(0)
            visual_feature = self.visual_pool_layer(visual_feature)
            textual_feature = self.textual_pool_layer(textual_feature)

            visual_embed = visual_feature.view(batch_size, -1)
            textual_embed = textual_feature.view(batch_size, -1)

            visual_embed = self.visual_embed_layer(visual_embed)
            textual_embed = self.textual_embed_layer(textual_embed)
        elif self.embed_style == 'multi':
            # visual branch, resnet backbone
            _, _, layer3, layer4 = visual_feature
            h = layer4.size(2)
            assert h % self.branch_size == 0, f"height {h} is not divided by {self.branch_size}."
            part_h = h // self.branch_size
            visual_embed = []
            img_f3 = self.visual_pool_layer(self.visual_embed_layer_lower(layer3)).squeeze(-1).squeeze(-1)
            visual_embed.append(img_f3)
            for ib in range(self.branch_size):
                visual_embed.append(
                    self.visual_pool_layer(
                        self.visual_embed_layer(layer4[:, :, part_h * ib:part_h * (ib + 1), :])).squeeze(-1).squeeze(-1)
                )
            img_f4 = self.visual_pool_layer(self.visual_embed_layer(layer4)).squeeze(-1).squeeze(-1)
            if self.bnneck_global:
                img_f4 = self.bottleneck_image(img_f4)
            visual_embed.append(img_f4)
            # text branch, bert backbone
            textual_embed = []
            textual_feature = textual_feature.unsqueeze(1).permute(0, 3, 1, 2).contiguous()
            text = self.textual_embed_layer(textual_feature)
            text_f3 = self.textual_pool_layer(text).squeeze(-1).squeeze(-1)
            textual_embed.append(text_f3)
            for idx_h in range(self.branch_size):
                if self.text_branches_type == 'conv':
                    textual_embed.append(
                        self.textual_pool_layer(self.text_branches[idx_h](text)).squeeze(-1).squeeze(-1)
                    )
                elif self.text_branches_type == 'transformer':
                    padding_mask = ~kwargs['text_mask'].to(torch.bool) if 'text_mask' in kwargs else None
                    text_ = text.squeeze(2).permute(2, 0, 1).contiguous()
                    textual_attn_output, textual_attn_output_weights = self.text_branches[idx_h](text_, text_, text_,
                                                                                                 key_padding_mask=padding_mask)
                    textual_embed.append(textual_attn_output[0])  # use the first one
                else:
                    raise NotImplementedError(f'{self.text_branches_type} is not supported.')
            text_f4_stack = torch.stack(textual_embed[1:], dim=2).unsqueeze(-1)  # stack part-based features
            text_f4 = self.textual_pool_layer(text_f4_stack).squeeze(-1).squeeze(-1)
            if self.bnneck_global:
                text_f4 = self.bottleneck_text(text_f4)
            textual_embed.append(text_f4)
        elif self.embed_style == 'mha':
            # visual_feature (B, S+1, C), textual_feature (B, L+1, C)
            visual_attn_output, visual_attn_output_weights = self.visual_embed_layer(visual_feature, visual_feature,
                                                                                     visual_feature)
            visual_embed = torch.chunk(visual_attn_output[:, :, 0], chunks=self.num_heads, dim=1)
            visual_embed = [embed.squeeze(dim=1) for embed in visual_embed]

            padding_mask = ~kwargs['text_mask'].to(torch.bool) if 'text_mask' in kwargs else None
            textual_attn_output, textual_attn_output_weights = self.textual_embed_layer(textual_feature, textual_feature,
                                                                                       textual_feature,
                                                                                       k_mask=padding_mask)
            textual_embed = torch.chunk(textual_attn_output[:, :, 0], chunks=self.num_heads, dim=1)
            textual_embed = [embed.squeeze(dim=1) for embed in textual_embed]

            visual_embed_global = visual_feature[:, 0]
            textual_embed_global = textual_feature[:, 0]
            if self.bnneck_global:
                visual_embed_global = self.bottleneck_image(visual_embed_global)
                textual_embed_global = self.bottleneck_text(textual_embed_global)

            visual_embed.append(visual_embed_global)
            textual_embed.append(textual_embed_global)
        else:
            raise NotImplementedError('Unsupported embed style {}'.format(self.embed_style))

        # feed embeddings into the invertible neural network
        v2t_embed, v2t_all_states = self.invertible_net(visual_embed)
        t2v_embed, t2v_all_states = self.invertible_net.reverse(textual_embed)

        bn_kwargs = {}
        # if enable the classification/instance loss
        if self.bnneck:
            if self.embed_style == 'single':
                visual_embed_bn = self.visual_bnneck(visual_embed)
                textual_embed_bn = self.textual_bnneck(textual_embed)
                t2v_embed_bn = self.visual_bnneck(t2v_embed)
                v2t_embed_bn = self.textual_bnneck(v2t_embed)
            elif self.embed_style in ['multi', 'mha']:
                # only use the last global visual and textual embedding
                visual_embed_bn = self.visual_bnneck(visual_embed[-1])
                textual_embed_bn = self.textual_bnneck(textual_embed[-1])
                if self.invert_style == 'transformer':
                    t2v_embed_bn = self.visual_bnneck(t2v_embed[:, -1])
                    v2t_embed_bn = self.textual_bnneck(v2t_embed[:, -1])
                else:
                    t2v_embed_bn = self.visual_bnneck(t2v_embed[-1])
                    v2t_embed_bn = self.textual_bnneck(v2t_embed[-1])
            else:
                raise NotImplementedError('Unsupported embed style {}'.format(self.embed_style))

            visual_embeds_bn = [visual_embed_bn, t2v_embed_bn]
            textual_embeds_bn = [textual_embed_bn, v2t_embed_bn]
            bn_kwargs.update({'visual_embeds_bn': visual_embeds_bn, 'textual_embeds_bn': textual_embeds_bn})

        if self.training:
            losses = self.loss_evaluator(v2t_all_states, t2v_all_states, captions, **bn_kwargs)
            return None, losses

        # output for testing
        outputs = self.construct_output(v2t_all_states, t2v_all_states)
        return outputs, None

    def construct_output(self, v2t_all_states, t2v_all_states):
        outputs = []
        if self.output_all_scales and self.embed_style in ['multi', 'mha']:
            out1, out2 = [], []
            if self.invert_style == 'transformer':
                v2t_all_states = [[t[:, i] for i in range(t.size(1))] for t in v2t_all_states]
                t2v_all_states = [[v[:, i] for i in range(v.size(1))] for v in t2v_all_states]
            if self.output_all_states:
                for idx in range(len(v2t_all_states)):  # different RevNet
                    out1.extend(v2t_all_states[idx])
                    out2.extend(list(reversed(t2v_all_states[idx])))
            else:
                for idx in range(len(v2t_all_states)):
                    out1.extend([v2t_all_states[idx][0], v2t_all_states[idx][-1]]) # use input and output of each end
                    out2.extend([t2v_all_states[idx][-1], t2v_all_states[idx][0]])
            outputs.append(out1)
            outputs.append(out2)
        else:
            # use the last layer's embeddings (global feature) if multi head RevNet features are extracted
            if self.embed_style in ['multi', 'mha']:
                v2t_all_states = v2t_all_states[-1]
                t2v_all_states = t2v_all_states[-1]

            if self.output_all_states:
                outputs.append(v2t_all_states)
                outputs.append(list(reversed(t2v_all_states)))
            else:
                outputs.append(v2t_all_states[0])
                outputs.append(t2v_all_states[-1])
                # outputs.append([v2t_all_states[0], v2t_all_states[-1]])
                # outputs.append([t2v_all_states[-1], t2v_all_states[0]])
        return outputs


def build_invertible_head(cfg, visual_size, textual_size):
    model = InvertHead(cfg, visual_size, textual_size)
    return model
