import torch
import torch.nn as nn
from torch.nn import init

from .loss import make_loss_evaluator


class SeqAttention(nn.Module):
    def __init__(self, num_features=128, k=3, seqlen=64):
        super(SeqAttention, self).__init__()
        self.seqlen = seqlen
        self.k = k
        self.num_features = num_features

        self.conv1 = nn.Conv1d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(64, self.k, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(self.k)

        self.softmax = nn.Softmax(dim=-1)
        self.reset_params()

    def forward(self, x, mask=None):
        assert x.size()[-1] == self.seqlen, f"x.size()[-1]({x.size()[-1]}) not match seqlen({self.seqlen})"

        atn = self.conv1(x)
        atn = self.bn1(atn)
        atn = self.relu1(atn)
        atn = self.conv2(atn)
        atn = self.bn2(atn)
        atn = atn.view(-1, self.k, self.height * self.width)
        atn = self.softmax(atn)
        return atn

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


class SpaAttention(nn.Module):
    def __init__(self, num_features=128, k=3, height=12, width=4):
        super(SpaAttention, self).__init__()
        self.height = height
        self.width = width
        self.k = k
        self.num_features = num_features

        self.conv1 = nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, self.k, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.k)

        self.softmax = nn.Softmax(dim=-1)
        self.reset_params()

    def forward(self, x):
        assert x.size()[-2:] == (self.height, self.width), f"x.size()[-2:]({x.size()[-2:]}) not match (heigth, width)({self.height, self.width})"

        atn = self.conv1(x)
        atn = self.bn1(atn)
        atn = self.relu1(atn)
        atn = self.conv2(atn)
        atn = self.bn2(atn)
        atn = atn.view(-1, self.k, self.height * self.width)
        atn = self.softmax(atn)
        return atn

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

class KAttHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        self.spa_att_net = SpaAttention(cfg.MODEL.KATT.NUM_EMBED,
                                        cfg.MODEL.KATT.K,
                                        cfg.MODEL.KATT.SPA_HEIGHT,
                                        cfg.MODEL.KATT.SPA_WIDTH)
        self.text_att_net = SeqAttention(cfg.MODEL.KATT.NUM_EMBED,
                                         cfg.MODEL.KATT.K,
                                         cfg.MODEL.KATT.SEQLEN)

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
        import pdb
        pdb.set_trace()
        _, _, layer3, layer4 = self.visual_embed_layer(visual_feature)
        img_f3 = self.maxpool(layer3).squeeze(-1).squeeze(-1)

        att_visual = self.spa_att_net(layer4)
        img_f4_att = att_visual * layer4
        img_f4_list = torch.split(img_f4_att, dim=2)
        img_f4 = self.maxpool(layer4).squeeze(-1).squeeze(-1)
        visual_embeds = [img_f3]
        visual_embeds.extend(img_f4_list).extend(img_f4)

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
        
        att_textual = self.text_att_net(textual_embed)

        if self.training:
            losses = self.loss_evaluator(visual_embeds, textual_embed, captions)
            return None, losses

        if self.training:
            losses = self.loss_evaluator(visual_embeds, text_embeds, captions)
            return None, losses

        outputs = list()
        outputs.append(img_f4)
        outputs.append(text_f4)
        return outputs, None


def build_k_att_head(cfg, visual_size, textual_size):
    model = KAttHead(cfg, visual_size, textual_size)
    return model
