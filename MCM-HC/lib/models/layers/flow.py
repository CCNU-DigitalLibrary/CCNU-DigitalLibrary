import sys
import torch
import torch.nn as nn
import queue
from functools import partial

from .layer_norm import LayerNorm
from .misc import *
from .multihead_attention import MultiheadAttention, MultiheadRelativeAttention


class MSRevSymmCouplingFlow(nn.Module):
    def __init__(self, cfg, num_ends=None):
        super(MSRevSymmCouplingFlow, self).__init__()
        self.num_ends = num_ends if num_ends else cfg.MODEL.INVERT.EMBEDDING.NUM_BRANCHES + 2
        self.multi_scale_ends = nn.ModuleList(
            RevSymmCouplingFlow(cfg) for _ in range(self.num_ends))

    def forward(self, x, reverse=False):
        assert len(x) == self.num_ends, f'number of features({len(x)}) is not match number of ends({self.num_ends})'
        out = []
        all_states = []
        for idx_branch in range(self.num_ends):
            if not reverse:
                end_out, end_all_states = self.multi_scale_ends[idx_branch](x[idx_branch])
                out.append(end_out)
                all_states.append(end_all_states)
            else:
                end_out, end_all_states = self.multi_scale_ends[idx_branch].reverse(x[idx_branch])
                out.append(end_out)
                all_states.append(end_all_states)
        return out, all_states

    def reverse(self, out):
        return self.forward(out, reverse=True)


class RevSymmCouplingFlow(nn.Module):
    """
    symmetric models
    """

    def __init__(self, cfg):
        super().__init__()
        self.ends = nn.ModuleDict({
            'src': RevCouplingEnd(cfg),
            'tgt': RevCouplingEnd(cfg),
        })
        self.norms = nn.ModuleDict({
            'src': ActNorm(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.INVERT.INVERT_NET.INIT_ACTNORM),
            'tgt': ActNorm(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.INVERT.INVERT_NET.INIT_ACTNORM),
        })

    def forward(self, x, reverse=False):
        if isinstance(x, list):
            x = torch.stack(x, dim=1)  # B x T x C
        src_end = self.ends['src']
        tgt_end = self.ends['tgt']
        src_norm = self.norms['src']
        tgt_norm = self.norms['tgt']
        all_states = [x]
        if not reverse:
            # in norm
            x_norm = src_norm(x)
            all_states.append(x_norm)
            # src_end -> mid
            mid, src_end_states = src_end(x_norm)
            all_states.extend(src_end_states)
            # mid -> tgt_end
            out, tgt_end_states = tgt_end.reverse(mid)
            all_states.extend(tgt_end_states)
            # out norm
            out_norm = tgt_norm.reverse(out)
            all_states.append(out_norm)
        else:
            # out norm
            x_norm = tgt_norm(x)
            all_states.append(x_norm)
            # tgt_end -> mid
            mid, tgt_end_states = tgt_end(x_norm)
            all_states.extend(tgt_end_states)
            # mid -> src_end
            out, src_end_states = src_end.reverse(mid)
            all_states.extend(src_end_states)
            # in norm
            out_norm = src_norm.reverse(out)
            all_states.append(out_norm)
        return out_norm, all_states

    def reverse(self, out):
        return self.forward(out, reverse=True)


class RevCouplingEnd(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.n_flows = cfg.MODEL.INVERT.INVERT_NET.NUM_FLOWS
        self.mid_channels = cfg.MODEL.INVERT.INVERT_NET.HIDDEN_DIM
        self.num_blocks = cfg.MODEL.INVERT.INVERT_NET.HIDDEN_DEPTH
        self.feature_shuffle = cfg.MODEL.INVERT.INVERT_NET.FEAT_SHUFFLE
        self.norm_type = cfg.MODEL.INVERT.INVERT_NET.NORM_TYPE
        self.sub_layers = nn.ModuleList()
        self.init_actnorm = cfg.MODEL.INVERT.INVERT_NET.INIT_ACTNORM

        for flow in range(self.n_flows):
            self.sub_layers.append(DoubleCouplingFlowBlock(
                self.in_channels,
                self.mid_channels,
                self.num_blocks,
                self.feature_shuffle,
                self.norm_type,
                self.init_actnorm))

    def forward(self, x, reverse=False):
        # B x C
        mid_states = []
        if not reverse:
            for i in range(self.n_flows):
                x = self.sub_layers[i](x)
                mid_states.append(x)
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i].reverse(x)
                mid_states.append(x)
        return x, mid_states

    def reverse(self, out):
        return self(out, reverse=True)


class RevPureAffineCouplingEnd(RevCouplingEnd):
    """Flat, multiple blocks of DoubleAffineCoupling"""

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.sub_layers
        self.sub_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(PureAffineDoubleCouplingFlowBlock(
                self.in_channels, self.mid_channels,
                self.num_blocks,
                self.init_actnorm))


class DoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth, feature_shuffle, norm_type='batchnorm', init_actnorm='gaussian'):
        super().__init__()
        self.norm_layer = ActNorm(in_channels, init_actnorm)
        self.coupling = DoubleVectorCouplingBlock(in_channels,
                                                  hidden_dim,
                                                  hidden_depth,
                                                  norm_type,
                                                  init_actnorm)
        self.shuffle = None
        if feature_shuffle:
            self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            x = self.norm_layer(x)
            x = self.coupling(x)
            if self.shuffle is not None:
                x = self.shuffle(x)
        else:
            if self.shuffle is not None:
                x = self.shuffle.reverse(x)
            x = self.coupling.reverse(x)
            x = self.norm_layer.reverse(x)
        return x

    def reverse(self, out):
        return self.forward(out, reverse=True)


class PureAffineDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth, init_actnorm):
        super().__init__()
        self.coupling = DoubleVectorCouplingBlock(in_channels,
                                                  hidden_dim,
                                                  hidden_depth,
                                                  init_actnorm)

    def forward(self, x, reverse=False):
        if not reverse:
            h = self.coupling(x)
        else:
            h = self.coupling(x, reverse=True)
        return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class DoubleVectorCouplingBlock(nn.Module):
    """Support uneven inputs"""

    def __init__(self, in_channels, hidden_dim, hidden_depth=2, norm_type='batchnorm', init_actnorm='gaussian'):
        super().__init__()
        dim1 = (in_channels // 2) + (in_channels % 2)
        dim2 = in_channels // 2
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True, norm_type=norm_type, init_actnorm=init_actnorm),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True, norm_type=norm_type, init_actnorm=init_actnorm),
        ])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False, norm_type=norm_type, init_actnorm=init_actnorm),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False, norm_type=norm_type, init_actnorm=init_actnorm),
        ])

    def forward(self, x, reverse=False):
        if not reverse:
            idx_apply, idx_keep = 0, 1
            for i in range(len(self.s)):
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * scale.exp() + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x

    def reverse(self, x):
        return self.forward(x, reverse=True)


class RevTransformerFlow(RevSymmCouplingFlow):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ends = nn.ModuleDict({
            'src': RevTransformerEnd(cfg),
            'tgt': RevTransformerEnd(cfg)
        })
        self.norms = nn.ModuleDict({
            'src': ActNorm(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.INVERT.INVERT_NET.INIT_ACTNORM),
            'tgt': ActNorm(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.INVERT.INVERT_NET.INIT_ACTNORM),
        })


class RevTransformerEnd(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_flows = cfg.MODEL.INVERT.INVERT_NET.NUM_FLOWS
        self.sub_layers = nn.ModuleList(
            [RevTransformerFlowBlock(cfg) for _ in range(self.n_flows)]
        )

    def forward(self, x, reverse=False):
        # B x S x C -> S x B x C
        x = x.transpose(0, 1)
        mid_states = []
        if not reverse:
            for i in range(self.n_flows):
                x = self.sub_layers[i](x)
                mid_states.append(x.transpose(0, 1))
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i].reverse(x)
                mid_states.append(x.transpose(0, 1))
        # S x B x C -> B x S x C
        return x.transpose(0, 1), mid_states

    def reverse(self, out):
        return self(out, reverse=True)


class RevTransformerFlowBlock(nn.Module):
    """
    RevNet style Transformer layer
    Inspired by implementation of Reformer
    "Reformer: The Efficient Transformer" (https://github.com/lucidrains/reformer-pytorch)
    """
    RESIDUAL_FN_WRAPPERS = {
        "prenorm": partial(PreNorm, LayerNorm),
        "sandwich": partial(SandwichNorm, LayerNorm)
    }

    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.MODEL.EMBEDDING.FEATURE_SIZE // 2
        # self attention
        self.self_attention_type = cfg.MODEL.INVERT.INVERT_NET.TRANS.SELF_ATTNETION_TYPE
        self.num_heads = cfg.MODEL.INVERT.INVERT_NET.TRANS.NUM_HEADS
        self.att_dropout = cfg.MODEL.INVERT.INVERT_NET.TRANS.ATT_DROPOUT
        self.max_position = cfg.MODEL.INVERT.INVERT_NET.TRANS.MAX_POSITION
        # ffn
        self.layer_norm_type = cfg.MODEL.INVERT.INVERT_NET.TRANS.LAYER_NORM_TYPE
        self.encoder_ffn_dim = cfg.MODEL.INVERT.INVERT_NET.TRANS.ENCODER_FFN_DIM
        self.dropout = cfg.MODEL.INVERT.INVERT_NET.TRANS.DROPOUT

        self_attn = self.build_self_attention()
        self_attn.forward = lambda x, **kwargs: self_attn.__class__.forward(
            self_attn, query=x, key=x, value=x, **kwargs
        )

        feed_forward = self.build_ffn()

        residual_fn_wrapper = self.RESIDUAL_FN_WRAPPERS[self.layer_norm_type]
        self.self_attn = residual_fn_wrapper(
            self.in_channels, self.dropout, self_attn)
        self.feed_forward = residual_fn_wrapper(
            self.in_channels, self.dropout, feed_forward)

        # dropout requires to have the same
        # seed for forward and backward pass

        self.self_attn_seed = queue.Queue()
        self.feed_forward_seed = queue.Queue()
        self.forward_steps = 0
        self.reverse_steps = 0

    def build_self_attention(self):
        attn_cls = {
            'normal': MultiheadAttention,
            'relative': MultiheadRelativeAttention
        }[self.self_attention_type]

        return attn_cls(
            embed_dim=self.in_channels,
            num_heads=self.num_heads,
            dropout=self.att_dropout,
            self_attention=True,
            max_position=self.max_position
        )

    def build_ffn(self):
        return FeedForward(
            self.in_channels,
            self.encoder_ffn_dim,
            dropout=self.dropout,
            activation=nn.GELU
        )

    def _init_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(seed)
        return seed

    def f(self, prev_hidden_states, self_attn_padding_mask=None, self_attn_mask=None, rel_attn_kv=None):
        # self attention
        if self.forward_steps == self.reverse_steps:
            self.self_attn_seed.put(self._init_seed())
        else:
            torch.manual_seed(self.self_attn_seed.get())

        self_attn_output, attn_weights = self.self_attn(
            prev_hidden_states,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            rel_attn_kv=rel_attn_kv
        )
        return self_attn_output, attn_weights

    def g(self, attn_output):
        if self.forward_steps == self.reverse_steps:
            self.feed_forward_seed.put(self._init_seed())
        else:
            torch.manual_seed(self.feed_forward_seed.get())
        return self.feed_forward(attn_output)

    def forward(self, x, encoder_padding_mask=None, attn_mask=None, rel_attn_kv: list = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # every forward pass we sample a different seed
        # for dropout and save for forward fn in backward pass
        # to have correct dropout

        """ X_1,            X_2 """
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # Y_1 = X_1 + f(X_2)
        # f(X_2) = self_attn(X_2)
        fx2, attn_weights = self.f(
            prev_hidden_states=x2,
            self_attn_padding_mask=encoder_padding_mask,
            self_attn_mask=attn_mask,
            rel_attn_kv=rel_attn_kv,
        )
        """ Y_1 = X_1 + f(X_2) """
        y1 = x1 + fx2

        # # free memory
        # del prev_attn_output

        # Y_2 = X_2 + g(Y_1)
        # g(Y_1) = FFN(Y_1)
        gy1 = self.g(y1)
        """ Y_2       =        X_2         +   g(Y_1) """
        y2 = x2 + gy1

        y = torch.cat([y1, y2], dim=-1)

        self.forward_steps += 1
        return y

    def reverse(self, y, encoder_padding_mask=None, attn_mask=None, rel_attn_kv: list = None):
        # every forward pass we sample a different seed
        # for dropout and save for forward fn in backward pass
        # to have correct dropout
        """ Y_1,            Y_2 """
        y1, y2 = torch.chunk(y, chunks=2, dim=-1)

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # Y_2 = X_2 + g(Y_1)
        # g(Y_1) = FFN(Y_1)
        gy1 = self.g(y1)
        """ X_2       =        Y_2         -   g(Y_1) """
        x2 = y2 - gy1

        # Y_1 = X_1 + f(X_2)
        # f(X_2) = self_attn(X_2)
        fx2, attn_weights = self.f(
            prev_hidden_states=x2,
            self_attn_padding_mask=encoder_padding_mask,
            self_attn_mask=attn_mask,
            rel_attn_kv=rel_attn_kv,
        )
        """ X_1 = Y_1 - f(X_2) """
        x1 = y1 - fx2

        # # free memory
        # del prev_attn_output

        x = torch.cat([x1, x2], dim=-1)

        self.reverse_steps += 1
        return x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
