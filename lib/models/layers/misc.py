import torch
import torch.nn as nn

__all__ = ['ActNorm',
           'BasicFullyConnectedNet',
           'Shuffle',
           'PreNorm',
           'SandwichNorm',
           'FeedForward']

# class ActNorm(nn.Module):
#     """ActNorm used in net2net."""
#     def __init__(self, num_features, affine=True,
#                  allow_reverse_init=True):
#         assert affine
#         super().__init__()
#         self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
#         self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
#         self.allow_reverse_init = allow_reverse_init
#
#         self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
#
#     def initialize(self, input):
#         with torch.no_grad():
#             flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
#             mean = (
#                 flatten.mean(1)
#                 .unsqueeze(1)
#                 .unsqueeze(2)
#                 .unsqueeze(3)
#                 .permute(1, 0, 2, 3)
#             )
#             std = (
#                 flatten.std(1)
#                 .unsqueeze(1)
#                 .unsqueeze(2)
#                 .unsqueeze(3)
#                 .permute(1, 0, 2, 3)
#             )
#
#             self.loc.data.copy_(-mean)
#             self.scale.data.copy_(1 / (std + 1e-8))
#
#     def forward(self, input, reverse=False):
#         if reverse:
#             return self.reverse(input)
#         _, _, height, width = input.shape
#
#         if self.training and self.initialized.item() == 0:
#             self.initialize(input)
#             self.initialized.fill_(1)
#
#         h = self.scale * (input + self.loc)
#         return h
#
#     def reverse(self, output):
#         if self.training and self.initialized.item() == 0:
#             if not self.allow_reverse_init:
#                 raise RuntimeError(
#                     "Initializing ActNorm in reverse direction is "
#                     "disabled by default. Use allow_reverse_init=True to enable."
#                 )
#             else:
#                 self.initialize(output)
#                 self.initialized.fill_(1)
#         h = output / self.scale - self.loc
#         return h


class ActNorm(nn.Module):
    """ActNorm used in REDER"""

    def __init__(self, in_features, init_actnorm='gaussian'):
        super().__init__()

        self.log_scale = nn.Parameter(torch.zeros(in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))
        self.init_actnorm = init_actnorm
        self.register_buffer("_initialized", torch.tensor(False))

    def initialize(self, input, reverse):
        # input: [bsz, d]
        with torch.no_grad():
            data = input.clone()
            if not reverse:
                self.log_scale.data = -data.detach().std(0, unbiased=False).log()
                self.bias.data = - data.detach().mean(0) / self.log_scale.exp()
            else:
                self.log_scale.data = data.detach().std(0, unbiased=False).log()
                self.bias.data = - data.detach().mean(0)

            self._initialized = torch.tensor(True)

    def forward(self, input, reverse=False):
        if not self._initialized:
            if self.init_actnorm == 'random':
                self.log_scale.data = torch.randn_like(self.log_scale.data)
                self.bias.data = torch.randn_like(self.bias.data)
            elif self.init_actnorm == 'gaussian':
                self.initialize(input, reverse)
            else:
                assert self.init_actnorm == 'none', f'{self.init_actnorm} is not supported'
        if not reverse:
            out = input * self.log_scale.exp() + self.bias
        else:
            out = (input - self.bias).div(self.log_scale.exp() + 1e-8)
        return out

    def reverse(self, output):
        return self.forward(output, reverse=True)


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, out_dim=None, norm_type='batchnorm', init_actnorm='gaussian'):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if norm_type == 'batchnorm':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'actnorm':
            layers.append(ActNorm(hidden_dim, init_actnorm))
        else:
            raise NotImplementedError("{norm_type} is not supported")
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == 'actnorm':
                layers.append(ActNorm(hidden_dim, init_actnorm))
            else:
                raise NotImplementedError("{norm_type} is not supported")
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...]
        else:
            return x[:, self.backward_shuffle_idx, ...]

    def reverse(self, out):
        return self.forward(out, reverse=True)



class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, dropout, fn):
        super().__init__()
        self.norm = norm_class(dim, elementwise_affine=False)
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        if isinstance(x, tuple):
            x = list(x)
            x[0] = self.dropout(x[0])
            return tuple(x)
        return self.dropout(x)


class SandwichNorm(nn.Module):
    def __init__(self, norm_class, dim, dropout, fn):
        super().__init__()
        self.prenorm = norm_class(dim)
        self.postnorm = norm_class(dim)
        self.fn = fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        x = self.prenorm(x)
        x = self.fn(x, **kwargs)
        if isinstance(x, tuple):
            x = list(x)
            x[0] = self.dropout(self.postnorm(x[0]))
            return tuple(x)
        return self.dropout(self.postnorm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, dim_inner=4, dropout=0.0, activation=nn.GELU, glu=False):
        super().__init__()
        self.glu = glu
        self.w1 = nn.Linear(dim, dim_inner * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim_inner, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
