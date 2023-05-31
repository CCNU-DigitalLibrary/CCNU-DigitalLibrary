import torch.nn as nn
import torch.nn.functional as F

from lib.models.layers.pooling import *
from lib.models.embeddings.basehead import Bottleneck, conv1x1, conv1x3
from .loss import make_loss_evaluator


class BaselineHead(nn.Module):
    def __init__(
            self,
            cfg,
            visual_size,
            textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK
        self.visual_pool = cfg.MODEL.EMBEDDING.VISUAL_POOL
        self.textual_pool = cfg.MODEL.EMBEDDING.TEXTUAL_POOL

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)

        downsample = nn.Sequential(
            conv1x1(self.embed_size * 2, self.embed_size),
            nn.BatchNorm2d(self.embed_size),
        )
        self.textual_embed_layer = nn.Sequential(
            conv1x1(textual_size, self.embed_size * 2),
            nn.BatchNorm2d(self.embed_size * 2),
            nn.ReLU(inplace=True),
            nn.Sequential(
                Bottleneck(inplanes=self.embed_size * 2, planes=self.embed_size, width=self.embed_size // 2,
                           downsample=downsample),
                Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=self.embed_size // 2),
                Bottleneck(inplanes=self.embed_size, planes=self.embed_size, width=self.embed_size // 2),
                # Bottleneck_with_rrelu(inplanes=self.embed_size * 2, planes=self.embed_size, width=self.embed_size // 2,
                #            downsample=downsample),
                # Bottleneck_with_rrelu(inplanes=self.embed_size, planes=self.embed_size, width=self.embed_size // 2),
                # Bottleneck_with_rrelu(inplanes=self.embed_size, planes=self.embed_size, width=self.embed_size // 2)
            )
        )
        # todo
        # convert the continue feature to hash code module

        # self.visual_linear = nn.Linear(512, 512)
        # self.textual_linear = nn.Linear(512, 512)

        self.visual_hash_module = nn.Sequential(
            nn.Linear(512, 512)
        )
        self.textual_hash_module = nn.Sequential(
            nn.Linear(512, 512)
        )
        #
        # self.visual_hash_module = nn.Sequential(
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 512)
        # )
        # self.textual_hash_module = nn.Sequential(
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 512)
        # )


        #self.visual_hash_module = nn.Sequential(
        #    nn.Linear(512, 2048, bias=True),
        #    nn.BatchNorm1d(2048),
        #    nn.ReLU(True),
        #    nn.Linear(2048, 512, bias=True),
        #    nn.Tanh()
        #)
        #self.textual_hash_module = nn.Sequential(
        #    nn.Linear(512, 2048, bias=True),
        #    nn.BatchNorm1d(2048),
        #    nn.ReLU(True),
        #    nn.Linear(2048, 512, bias=True),
        #    nn.Tanh()
        #)

        visual_pool_type = cfg.MODEL.EMBEDDING.VISUAL_POOL
        textual_pool_type = cfg.MODEL.EMBEDDING.TEXTUAL_POOL

        if visual_pool_type == 'fastavgpool':
            self.visual_pool_layer = FastGlobalAvgPool2d()
        elif visual_pool_type == 'avgpool':
            self.visual_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif visual_pool_type == 'maxpool':
            self.visual_pool_layer = nn.AdaptiveMaxPool2d(1)
        elif visual_pool_type == 'gempoolP':
            self.visual_pool_layer = GeneralizedMeanPoolingP()
        elif visual_pool_type == 'gempool':
            self.visual_pool_layer = GeneralizedMeanPooling()
        elif visual_pool_type == "avgmaxpool":
            self.visual_pool_layer = AdaptiveAvgMaxPool2d()
        elif visual_pool_type == 'clipavgpool':
            self.visual_pool_layer = ClipGlobalAvgPool2d()
        elif visual_pool_type == "identity":
            self.visual_pool_layer = nn.Identity()
        elif visual_pool_type == "flatten":
            self.visual_pool_layer = Flatten()
        else:
            raise KeyError(f"{visual_pool_type} is not supported!")

        if textual_pool_type == 'fastavgpool':
            self.textual_pool_layer = FastGlobalAvgPool2d()
        elif textual_pool_type == 'avgpool':
            self.textual_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif textual_pool_type == 'maxpool':
            self.textual_pool_layer = nn.AdaptiveMaxPool2d(1)
        elif textual_pool_type == 'gempoolP':
            self.textual_pool_layer = GeneralizedMeanPoolingP()
        elif textual_pool_type == 'gempool':
            self.textual_pool_layer = GeneralizedMeanPooling()
        elif textual_pool_type == "avgmaxpool":
            self.textual_pool_layer = AdaptiveAvgMaxPool2d()
        elif textual_pool_type == 'clipavgpool':
            self.textual_pool_layer = ClipGlobalAvgPool2d()
        elif textual_pool_type == "identity":
            self.textual_pool_layer = nn.Identity()
        elif textual_pool_type == "flatten":
            self.textual_pool_layer = Flatten()
        else:
            raise KeyError(f"{textual_pool_type} is not supported!")

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

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions, **kwargs):
        batch_size = visual_feature.size(0)

        visual_feature = self.visual_pool_layer(visual_feature)

        visual_embed = visual_feature.view(batch_size, -1)
        textual_embed = textual_feature.unsqueeze(1).permute(0, 3, 1, 2).contiguous()



        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_pool_layer(self.textual_embed_layer(textual_embed)).squeeze()

        # print(visual_embed.size())
        # print(textual_embed.size())
        # import pdb
        # pdb.set_trace()


        visual_embed = self.visual_hash_module(visual_embed)
        textual_embed= self.textual_hash_module(textual_embed)

        # print(visual_embed.size())
        # print(textual_embed.size())





        # print(self.textual_pool_layer)
        # print(sum)

        if self.shared_embed_layer is not None:
            visual_embed = self.shared_embed_layer(
                F.normalize(visual_embed, p=2, dim=1)
            )
            textual_embed = self.shared_embed_layer(
                F.normalize(textual_embed, p=2, dim=1)
            )

        if self.bnneck:
            print("self.bnneck")
            visual_embed_bn = self.visual_bnneck(visual_embed)
            textual_embed_bn = self.textual_bnneck(textual_embed)

            if self.training:
                losses = self.loss_evaluator(
                    visual_embed,
                    textual_embed,
                    captions,
                    visual_embed_bn,
                    textual_embed_bn,
                )
                return None, losses

            outputs = list()
            outputs.append(visual_embed_bn)
            outputs.append(textual_embed_bn)
            return outputs, None



        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)

        if self.training:
            # print("self.bnneck is false and self.training")
            losses = self.loss_evaluator(visual_embed, textual_embed, captions)

            return outputs, losses
        # losses = self.loss_evaluator(visual_embed, textual_embed, captions)



        return outputs, None


def build_baseline_head(cfg, visual_size, textual_size):
    model = BaselineHead(cfg, visual_size, textual_size)
    return model
