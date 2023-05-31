# encoding: utf-8
# TIPCB
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import make_loss_evaluator
from lib.models.embeddings.basehead import BaseHead, conv1x1, Bottleneck


class MultiscaleHead(BaseHead):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
        branch_size=6
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK
        self.branch_size = branch_size

        self.visual_embed = None
        if self.embed_size != visual_size:
            self.visual_embed = nn.Sequential(
                nn.Linear(visual_size, self.embed_size),
                nn.BatchNorm1d(self.embed_size),
                nn.ReLU(inplace=True)
            )

        self.text_embed = nn.Sequential(
            conv1x1(textual_size, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.build_text_branch()

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_embed_layer = None
        if cfg.MODEL.EMBEDDING.SHARED_LAYER:
            self.shared_embed_layer = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size),
            )

        if self.bnneck:
            self.visual_bnneck = nn.BatchNorm1d(self.embed_size)
            self.textual_bnneck = nn.BatchNorm1d(self.embed_size)
            self.visual_bnneck.bias.requires_grad_(False)  # no shift
            self.textual_bnneck.bias.requires_grad_(False)  # no shift

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def build_text_branch(self):
        self.text_branches = nn.ModuleList()
        for _ in range(self.branch_size):
            downsample = nn.Sequential(
                conv1x1(1024, self.embed_size),
                nn.BatchNorm2d(self.embed_size),
            )
            self.text_branches.append(
                nn.Sequential(
                    Bottleneck(inplanes=1024, planes=self.embed_size, width=512, downsample=downsample),
                    Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=512),
                    Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=512)
                    )
                )

    def forward(self, visual_feature, textual_feature, captions, **kwargs):
        # visual branch
        _, _, layer3, layer4 = visual_feature
        h = layer4.size(2)
        assert h % self.branch_size == 0, f"height {h} is not divided by {self.branch_size}."
        part_h = h // self.branch_size
        img_f3 = self.maxpool(layer3).squeeze(-1).squeeze(-1)
        img_f41 = self.maxpool(layer4[:, :, :part_h, :]).squeeze(-1).squeeze(-1)
        img_f42 = self.maxpool(layer4[:, :, part_h:2 * part_h, :]).squeeze(-1).squeeze(-1)
        img_f43 = self.maxpool(layer4[:, :, 2 * part_h:3 * part_h, :]).squeeze(-1).squeeze(-1)
        img_f44 = self.maxpool(layer4[:, :, 3 * part_h:4 * part_h, :]).squeeze(-1).squeeze(-1)
        img_f45 = self.maxpool(layer4[:, :, 4 * part_h:5 * part_h, :]).squeeze(-1).squeeze(-1)
        img_f46 = self.maxpool(layer4[:, :, 5 * part_h:, :]).squeeze(-1).squeeze(-1)
        img_f4 = self.maxpool(layer4).squeeze(-1).squeeze(-1)
        visual_embeds = [img_f3, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, img_f4]
        if self.visual_embed:
            visual_embeds = [img_f3, 
                            self.visual_embed(img_f41), 
                            self.visual_embed(img_f42), 
                            self.visual_embed(img_f43), 
                            self.visual_embed(img_f44), 
                            self.visual_embed(img_f45), 
                            self.visual_embed(img_f46), 
                            self.visual_embed(img_f4)
                            ]
        # text branch
        textual_feature = textual_feature.unsqueeze(1).permute(0, 3, 1, 2).contiguous()
        text = self.text_embed(textual_feature)
        text_f3 = self.maxpool(text).squeeze(-1).squeeze(-1)
        text_f41 = self.maxpool(self.text_branches[0](text)).squeeze(-1).squeeze(-1)
        text_f42 = self.maxpool(self.text_branches[1](text)).squeeze(-1).squeeze(-1)
        text_f43 = self.maxpool(self.text_branches[2](text)).squeeze(-1).squeeze(-1)
        text_f44 = self.maxpool(self.text_branches[3](text)).squeeze(-1).squeeze(-1)
        text_f45 = self.maxpool(self.text_branches[4](text)).squeeze(-1).squeeze(-1)
        text_f46 = self.maxpool(self.text_branches[5](text)).squeeze(-1).squeeze(-1)
        text_f4 = F.adaptive_max_pool1d(torch.stack([text_f41, text_f42, text_f43, text_f44, text_f45, text_f46], dim=2).contiguous(), output_size=(1,)).squeeze(-1)

        text_embeds = [text_f3, text_f41, text_f42, text_f43, text_f44, text_f45, text_f46, text_f4]

        if self.bnneck:
            visual_embed_bn = self.visual_bnneck(visual_embeds[-1])
            textual_embed_bn = self.textual_bnneck(text_embeds[-1])

            if self.training:
                losses = self.loss_evaluator(visual_embeds, text_embeds, captions, visual_embed_bn, textual_embed_bn)
                return None, losses

            outputs = list()
            outputs.append(visual_embed_bn)
            outputs.append(textual_embed_bn)
            return outputs, None

        if self.training:
            losses = self.loss_evaluator(visual_embeds, text_embeds, captions)
            return None, losses

        outputs = list()
        outputs.append(img_f4)
        outputs.append(text_f4)
        return outputs, None


def build_multiscale_head(cfg, visual_size, textual_size):
    model = MultiscaleHead(cfg, visual_size, textual_size)
    return model
