# encoding: utf-8
# cross modal losses
from .align_loss import *
from .cmpcm import *
from .coral_loss import *
from .diversity_loss import *
from .domain_loss import *
from .instance_loss import *
from .mask_loss import *
from .dual_softmax_loss import *
# intra modal losses
from .triplet_loss import *
from .contrastive_loss import contrastive_loss
from .utils import *


class BaseLossComputation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError