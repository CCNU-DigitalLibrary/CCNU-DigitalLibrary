# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Code for some of the transformers components in this file are initialized
# from their counterparts in Hugging Face Transformers library.

from typing import Callable, NamedTuple, Optional, Tuple, Union
from typing import Dict, Optional, Tuple, Union
from typing import Callable, List, Optional, Union
from typing import Any
from torch.nn import functional as F
from torch import nn, Tensor
import torch




def shift_dim(
    x: Tensor, src_dim: int = -1, dest_dim: int = -1, make_contiguous: bool = True
) -> Tensor:
    """Permutes tensor x by moving src_dim to dest_dim.
    i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

    Args:
        x (Tensor): input Tensor you want to permute
        src_dim (int, optional): the axis you want to move. Negative indexing supported. Defaults to -1.
        dest_dim (int, optional): the axis you want to move to. Negative indexing supported. Defaults to -1.
        make_contiguous (bool, optional): if you want the output tensor to be contiguous in memory. Defaults to True.

    Returns:
        Tensor: permuted Tensor
    """
    n_dims = len(x.shape)
    # Remap negative dim
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


class SelfAttention(nn.Module):
    """Computes attention over the entire n-dimensional input.

    Attributes:
        attn_dropout (float): probability of dropout after softmax. Default is ``0.0``.

    Args:
        q, k, v (Tensor): a [b, h, d1, ..., dn, c] tensor where h is the number of attention heads
        attention_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                           Contains 1s for positions to attend to and 0s for masked positions.
                                           Applied before softmax.
        head_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                      Contains 1s for positions to attend to and 0s for masked positions.
                                      Applied after dropout, before matrix multiplication with values.

    """

    def __init__(self, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        _, _, *shape, _ = q.shape

        # flatten to b, h, (d1, ..., dn), c
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out, attn_probs = scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            head_mask=head_mask,
            attn_dropout=self.attn_dropout if self.training else 0.0,
        )

        return out.unflatten(2, shape), attn_probs


class AxialAttention(nn.Module):
    """Computes attention over a single axis of the input. Other dims are flattened
    into the batch dimension.

    Attributes:
        axial_dim (int): dimension to compute attention on, index by input dimensions
            (i.e., 0 for first input dimension, 1 for second)
        attn_dropout (float): probability of dropout after softmax. Default is ``0.0``.

    Args:
        q, k, v (Tensor): a [b, h, d1, ..., dn, c] tensor where h is the number of attention heads
        attention_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                           Contains 1s for positions to attend to and 0s for masked positions.
                                           Applied before softmax.
        head_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                      Contains 1s for positions to attend to and 0s for masked positions.
                                      Applied after dropout, before matrix multiplication with values.

    """

    def __init__(self, axial_dim: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout
        self.axial_dim = axial_dim + 2  # account for batch, head

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Ensure axial dim is within right dimensions, should be between head dim and embedding dim
        if self.axial_dim >= len(q.shape) - 1:
            raise ValueError("axial dim does not match input shape")

        # flatten all dims into batch dimension except chosen axial dim and channel dim
        # b, h, d1, ..., dn, c -> (b, h, d1, ..., dn-1), axial_dim, c
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out, attn_probs = scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            head_mask=head_mask,
            attn_dropout=self.attn_dropout if self.training else 0.0,
        )
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out, attn_probs


class MultiHeadAttention(nn.Module):
    """Computes multihead attention with flexible attention mechanism and caching for fast decoding.

    Multihead attention linearly projects and divides queries, keys, and values into
    multiple 'heads'. This enables the computation of attention multiple times in
    parallel, creating more varied representations and allows the models to jointly
    attend to information from different representation subspaces at different positions,
    as described in Attention Is All You Need (Vaswani et al. 2017).

    Attributes:
        dim_q (int): dimensionality of query embedding vector
        dim_kv (int): dimensionality of key/value embedding vector
        n_head (int): number of attention heads
        attn_module (nn.Module): module of attention mechanism to use. Default is ``SelfAttention``.
                                 Should have interface of:
                                    (q: Tensor,
                                    k: Tensor,
                                    v: Tensor,
                                    attention_mask: Optional[Tensor],
                                    head_mask: Optional[Tensor],
                                    )
                                 and returns output Tensor and attn weights Tensor
        add_bias (bool): add bias to the q, k, v, linear layers or not. Default is ``True``.

    Args:
        q (Tensor): a tensor of shape [b, d1, ..., dn, c] or [b, seq_len, c]
            (for autoregressive decoding it's typical to pass in flattened tensors).
        kv (Optional[Tensor]): a tensor of shape [b, d1, ..., dn, c] or [b, seq_len, c].
                               If this argument is specified, this module become multiheaded cross-attention.
        attention_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                           Contains 1s for positions to attend to and 0s for masked positions.
                                           Applied before softmax.
        head_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                      Contains 1s for positions to attend to and 0s for masked positions.
                                      Applied after dropout, before matrix multiplication with values.
        use_cache (bool): If True, caches past k and v tensors for faster decoding. If False, recompute k and v for each
                          decoding step. Default is ``False``.
        causal (bool): use causal attention or not. Default is ``False``.

    Raises:
        TypeError: an error occurred when ``causal`` is ``True`` and ``attn_module`` is
            ``AxialAttention``.
        ValueError: when dim_q or dim_kv is not divisible by n_head
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_head: int,
        attn_module: nn.Module = SelfAttention(),
        add_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim_q % n_head != 0 or dim_kv % n_head != 0:
            raise ValueError(
                "The hidden size of q, k, v must be a multiple of the number of attention heads."
            )

        self.d_qk = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head
        self.query = nn.Linear(dim_q, n_head * self.d_qk, bias=add_bias)  # q
        self.key = nn.Linear(dim_kv, n_head * self.d_qk, bias=add_bias)  # k
        self.value = nn.Linear(dim_kv, n_head * self.d_v, bias=add_bias)  # v
        self.output = nn.Linear(n_head * self.d_v, dim_q, bias=True)  # c

        self.attn = attn_module

        self.cache: Optional[Dict[str, Tensor]] = None

    def forward(
        self,
        q: Tensor,
        kv: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        use_cache: bool = False,
        causal: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(self.attn, AxialAttention) and causal:
            raise TypeError("Causal axial attention is not supported.")

        # If kv is specified use those inputs for cross-attention, otherwise use q
        k = v = q if kv is None else kv
        # compute q
        q = split_multihead(self.query(q), self.n_head)

        # For causal k, v are provided step-wise so we should always compute them
        # For non-causal skip computing k, v if they have been cached
        if causal or not self.cache:
            k = split_multihead(self.key(k), self.n_head)
            v = split_multihead(self.value(v), self.n_head)

        # fast decoding by caching past key, value tensors
        if use_cache:
            if not self.cache:
                # initialize the cache with the present k, v
                self.cache = dict(k=k.clone(), v=v.clone())
            else:
                if causal:
                    # append present k, v to past k, v
                    # for autoregressive decoding inputs are flattened as 1D sequences
                    # so are the cached tensors: (b, n_heads, seq_len, c)
                    k_, v_ = self.cache["k"], self.cache["v"]
                    self.cache["k"] = torch.cat([k_, k], dim=2)
                    self.cache["v"] = torch.cat([v_, v], dim=2)
                # override the present k, v with the cache
                k, v = self.cache["k"], self.cache["v"]

        a, attn_probs = self.attn(q, k, v, attention_mask, head_mask)
        a = merge_multihead(a)
        a = self.output(a)

        if return_attn_weights:
            return a, attn_probs
        else:
            return a


class AxialAttentionBlock(nn.Module):
    """Computes multihead axial attention across all dims of the input.

    Axial attention is an alternative to standard full attention, where instead
    of computing attention across the entire flattened input, you compute it for
    each dimension. To capture the global context that full attention does, stacking
    multiple axial attention layers will allow information to propagate among the
    multiple dimensions of the input. This enables attention calculations on high
    dimensional inputs (images, videos) where full attention would be computationally
    expensive and unfeasible. For more details, see "Axial Attention in
    Multidimensional Transformers" (Ho et al. 2019) and CCNet (Huang et al. 2019).

    Follows implementation by VideoGPT:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Attributes:
        n_dims (int): dimensionality of input data, not including batch or channel dims
        qkv_dim (int): dimensionality of linear projections Wq, Wk, and Wv in attention
        n_head (int): number of heads in multihead attention. Must divide into qkv_dim
                      evenly

    Args:
        x (Tensor): a [b, c, d1, ..., dn] tensor, where c == qkv_dim
    """

    def __init__(self, n_dims: int, qkv_dim: int, n_head: int) -> None:
        super().__init__()
        self.qkv_dim = qkv_dim
        self.mha_attns = nn.ModuleList(
            [
                MultiHeadAttention(
                    dim_q=qkv_dim,
                    dim_kv=qkv_dim,
                    n_head=n_head,
                    attn_module=AxialAttention(d),
                    add_bias=False,
                )
                for d in range(n_dims)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        n_channel = x.shape[1]
        if n_channel != self.qkv_dim:
            raise ValueError(
                f"Input channel dimension is {n_channel}, expected {self.qkv_dim}"
            )

        h = shift_dim(x, 1, -1)  # (b, c, d1, ..., dn) -> (b, d1, ..., dn, c)
        attn_out = torch.zeros_like(h)
        for mha_attn in self.mha_attns:
            attn_out += mha_attn(h)
        h = attn_out
        h = shift_dim(h, -1, 1)  # (b, d1, ..., dn, c) -> (b, c, d1, ..., dn)
        return h



class MLP(nn.Module):
    """A multi-layer perceptron module.

    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]] = None,
        dropout: float = 0.5,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    attn_dropout: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Similar to PyTorch Core's _scaled_dot_product_attention but generalized
    to handle n-dimensional input tokens (images, video) and support multihead.
    Computes attention as described in Attention Is All You Need (Vaswani et al. 2017)

    Args:
        q, k, v (Tensor): a [b, h, d1, ..., dn, c] tensor or a flattened tensor of shape [b, seq_len, c]
                          where first dim is batch dim and last dim is channel dim
        attention_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                           Contains 1s for positions to attend to and 0s for masked positions.
                                           Applied before softmax.
        head_mask (Optional[Tensor]): Tensor of shape [b, h, d1, ..., q_dn, k_dn].
                                      Contains 1s for positions to attend to and 0s for masked positions.
                                      Applied after dropout, before matrix multiplication with values.
        attn_dropout (float): probability of dropout after softmax. Default is ``0.0``.
    """

    # Take the dot product between "query" and "key" and scale to get the raw attention scores.
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1]))
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor with the computed attention weights
    # at the positions we want to attend and -inf for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    if attention_mask is not None:
        attn = attn.masked_fill(attention_mask == 0, float("-inf"))
    # Normalize the attention scores to probabilities
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b, h, d1, ..., q_dn, k_dn
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn = F.dropout(attn, p=attn_dropout)
    # Mask heads if we want to
    if head_mask is not None:
        attn = attn * head_mask
    # For each query sum over the key/value dim with attention weights
    a = torch.matmul(attn, v)  # b, h, d1, ..., q_dn, c

    return a, attn


def split_multihead(x: Tensor, n_head: int) -> Tensor:
    """Splits channel dimension of input tensor of size (b, d1, ..., dn, c)
    into multiple heads, (b, n_head, d1, ..., dn, c // n_head)"""
    x = x.unflatten(-1, (n_head, -1))
    # Rearrange to put head dim first, (b, n_head, d1, ..., dn, c // n_head)
    x = shift_dim(x, -2, 1)
    return x


def merge_multihead(x: Tensor) -> Tensor:
    """Moves head dim back to original location and concatenates heads
    (b, n_head, d1, ..., dn, c // n_head) -> (b, d1, ..., dn, c)"""
    return shift_dim(x, 1, -2).flatten(start_dim=-2)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class TransformerOutput(NamedTuple):
    last_hidden_state: Optional[Tensor] = None
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
    image_labels: Optional[Tensor] = None


class TransformerCrossAttentionLayer(nn.Module):
    """Transformer layer with self-attention on inputs and cross-attention on an encoder's outputs.
    Can be used in a transformer decoder or an encoder with cross-attention. Similar to
    ``nn.TransformerDecoderLayer``, but generalized for use in an encoder with cross-attention as well.
    Uses a custom ``MultiHeadAttention`` that supports n-dimensional inputs including sequences,
    images, video.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        encoder_hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate
            cross-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``.
            See ``MultiHeadAttention`` for shape requirements.
        cross_attention_mask (Tensor, optional): mask to be applied to cross-attention inputs,
            ``encoder_hidden_states``. See ``MultiHeadAttention`` for shape requirements.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        # attention block
        self.attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)
        # cross attention block
        self.cross_attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.cross_attention_dropout = nn.Dropout(dropout)
        # feedforward block
        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        output = self.attention(
            hidden_states, attention_mask=attention_mask, return_attn_weights=False
        )
        output = self.attention_dropout(output)
        return output

    def _cross_attention_block(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.cross_attention(
            hidden_states,
            encoder_hidden_states,
            attention_mask=cross_attention_mask,
            return_attn_weights=False,
        )
        output = self.cross_attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._self_attention_block(inputs, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.cross_attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(
            attn_norm_output, kv, cross_attention_mask
        )
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.feedforward_layernorm(
            cross_attention_residual
        )
        ff_residual = cross_attention_norm_output + self._feedforward_block(
            cross_attention_norm_output
        )
        return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        attn_output = self._self_attention_block(x, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(
            attn_norm_output, kv, cross_attention_mask
        )
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.cross_attention_layernorm(
            cross_attention_residual
        )
        ff_residual = cross_attention_norm_output + self._feedforward_block(
            cross_attention_norm_output
        )
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_first:
            return self._forward_prenorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer is made up of multihead self-attention and feedforward blocks,
    based on the architecture in "Attention Is All You Need" (Vaswani et al. 2017). Similar to
    ``nn.TransformerEncoderLayer``, but uses a custom ``MultiHeadAttention`` that supports
    n-dimensional inputs (including sequences, images, video) and head-masking.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``. See
            ``MultiHeadAttention`` for shape requirements.
        head_mask (Tensor, optional): mask to be applied to self-attention inputs after softmax and dropout,
            before matrix multiplication with values. See ``MultiHeadAttention`` for shape requirements.
        return_attn_weights (bool, optional): return attention probabilities in addition to attention output.
            Defaults to False.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        # attention block
        self.attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)
        # feedforward block
        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        output, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_attn_weights=True,
        )
        output = self.attention_dropout(output)
        return output, attn_weights

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        inputs = self.attention_layernorm(x)
        attn_output, attn_weights = self._attention_block(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(
            self.feedforward_layernorm(attn_residual)
        )
        if return_attn_weights:
            return ff_residual, attn_weights
        else:
            return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        attn_output, attn_weights = self._attention_block(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(
            self.attention_layernorm(attn_residual)
        )
        outputs = self.feedforward_layernorm(ff_residual)
        if return_attn_weights:
            return outputs, attn_weights
        else:
            return outputs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.norm_first:
            return self._forward_prenorm(
                hidden_states,
                attention_mask,
                head_mask,
                return_attn_weights,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                attention_mask,
                head_mask,
                return_attn_weights,
            )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_head,
                    dim_feedforward,
                    dropout,
                    activation,
                    layer_norm_eps,
                    norm_first,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        all_hidden_states: Tuple[Tensor, ...] = () if return_hidden_states else None
        all_self_attentions: Tuple[Tensor, ...] = () if return_attn_weights else None

        for layer_module in self.layer:
            if return_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                return_attn_weights=return_attn_weights,
            )

            if return_attn_weights:
                hidden_states = layer_outputs[0]
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

        if return_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class Pooler(nn.Module):
    def __init__(self, hidden_size: int = 768, **kwargs: Any):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the models by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def flava_multimodal_encoder(
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        dropout: float = 0.0,
        intermediate_size: int = 3072,
        intermediate_activation: Callable[..., nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-12,
):
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        activation=intermediate_activation,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        norm_first=True,
    )
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return encoder, layernorm, pooler


if __name__ == '__main__':
    pass
    # def flava_multimodal_encoder(
    #         hidden_size: int = 768,
    #         num_attention_heads: int = 12,
    #         num_hidden_layers: int = 12,
    #         dropout: float = 0.0,
    #         intermediate_size: int = 3072,
    #         intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    #         layer_norm_eps: float = 1e-12,
    # ):
    #     encoder = TransformerEncoder(
    #         n_layer=num_hidden_layers,
    #         d_model=hidden_size,
    #         n_head=num_attention_heads,
    #         dim_feedforward=intermediate_size,
    #         activation=intermediate_activation,
    #         layer_norm_eps=layer_norm_eps,
    #         dropout=dropout,
    #         norm_first=True,
    #     )
    #     layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    #     pooler = Pooler(hidden_size=hidden_size)
    #
    #     return encoder, layernorm, pooler
    #
    # mm_encoder, mm_layernorm, pooler = flava_multimodal_encoder()
    #
    # visual_feature = torch.randn(64, 192, 768)
    # textual_feature = torch.randn(64, 64, 768)
    # mm_feature = torch.cat([textual_feature, visual_feature], dim=1)
    # print(mm_feature.size())
    #
    # mm_feature_output = mm_encoder(mm_feature)
    # print(mm_feature_output.last_hidden_state.size())
    #
