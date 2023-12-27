import torch
import torch.nn as nn
import numpy as np
from transformers import Conv1D
from opt_einsum import contract
from typing import Optional, Tuple, Union
import layers


def _kn_pr_conv1d(x, r, p_inv, q, use_p, use_q, A, B, b, m1, n1, m2, n2):
    shape = x.size()
    if use_p:
        x = x[:, :, p_inv]
    x = x.view((-1, m2))
    Bx = contract("km,rmn->rkn", x, B)
    Bx = Bx.view((r, np.prod(shape[:-1]), m1, n2))  # (r,batch,m1,n2)
    ABx = contract("rnm,rbmj->bnj", A, Bx)  # (batch,n1,n2)

    ABx = torch.reshape(ABx, shape[:-1] + (n1 * n2,))
    if use_q:
        ABx = ABx[:, :, q]
    ABx = ABx + b
    return ABx


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class kn_pr_Conv1D(nn.Module):
    """
    Args:
        nf = n1 * n2 (`int`): The number of output features.
        nx = m1 * m2 (`int`): The number of input features.

        This layer provide alternative of linear layer. Weight decompose into
        kronecker with permutation decomposition.
        r - rank, (m1,n1,m2,n2) - shapes of matrixes A,B
        use_p - use permutation p(of rows)
        use_q - use permutation q(of columns)

        Example:
        >>>import kn_transformers
        >>>linear_layer = nn.Linear(2048,8192)
        >>>linear_layer = kn_transformers.layers.kn_pr_Conv1D.get_from_layer(linear_layer,
        >>>                                                                  4, 1024, 2048, 2, 4,
        >>>                                                                  verbose = True,n_iter_als = -1,
        >>>                                                                  n_iter = 5,
        >>>                                                                  method_type = 'random',use_p = True,use_q = True)
    """

    def __init__(self, r, m1, n1, m2, n2, use_p=True, use_q=True):
        super().__init__()
        A = torch.empty((r, n1, m1))
        B = torch.empty((r, m2, n2))
        nn.init.normal_(A, std=0.001)
        nn.init.normal_(B, std=0.001)

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(torch.zeros(n1 * n2))
        self.p = torch.randperm(m1 * m2)
        q = torch.randperm(n1 * n2)
        self.register_buffer("q", q)
        self.m1 = m1
        self.n1 = n1
        self.m2 = m2
        self.n2 = n2
        self.r = r
        self.use_p = use_p
        self.use_q = use_q
        p_inv = inverse_permutation(self.p)
        self.register_buffer("p_inv", p_inv)

    @classmethod
    def get_from_layer(
        cls,
        layer,
        r,
        m1,
        n1,
        m2,
        n2,
        use_p=True,
        use_q=True,
    ):
        params_ = []
        for name, param in layer.named_parameters():
            params_.append(param)
        w = params_[0]
        b = params_[1]
        if isinstance(layer, nn.Linear):
            w = w.t()
        w = w.detach()
        nx = w.shape[0]
        nf = w.shape[1]
        assert nx == m1 * m2
        assert nf == n1 * n2
        kn_p_layer = cls(r, m1, n1, m2, n2, use_p=use_p, use_q=use_q)
        kn_p_layer.q = torch.randperm(n1 * n2)
        kn_p_layer.p_inv = torch.randperm(m1 * m2)
        return kn_p_layer

    def forward(self, x):
        return _kn_pr_conv1d(
            x,
            r=self.r,
            p_inv=self.p_inv,
            q=self.q,
            use_p=self.use_p,
            use_q=self.use_q,
            A=self.A,
            B=self.B,
            b=self.b,
            m1=self.m1,
            n1=self.n1,
            m2=self.m2,
            n2=self.n2,
        )


def test_knpr(r, m1, n1, m2, n2, batch_size, seq_len):
    l = kn_pr_Conv1D(r, m1, n1, m2, n2)
    lc1 = Conv1D(n1 * n2, m1 * m2)
    A = torch.randn((r, n1, m1))
    B = torch.randn((r, m2, n2))
    l.b.data = lc1.bias
    l.A.data = A
    l.B.data = B
    p = torch.randperm(m1 * m2)
    p_inv = inverse_permutation(p)
    q = torch.randperm(n1 * n2)
    l.p_inv = p_inv
    l.q = q
    w = torch.kron(A[0].t().contiguous(), B[0].contiguous())
    for i in range(1, r):
        w += torch.kron(A[i].t().contiguous(), B[i].contiguous())
    lc1.weight.data = w[p, :][:, q]
    x = torch.randn((batch_size, seq_len, m1 * m2))
    print(torch.norm(l(x) - lc1(x)))


class KronAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
    ):
        super().__init__()

        attn_dec_shapes = config.attn_decomposition_shapes

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        assert self.head_dim * self.num_heads == self.embed_dim

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    # some parameters from the original implementation are removed as we don't
    # use cross-attention
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # cache is actually used in case you're wondering
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # neither do we reorder nor do we upcast attention
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

    @classmethod
    def from_gpt2_attn(
        cls,
        config,
        layer_idx: int,
        attn_weight_concat: torch.Tensor,
        attn_bias_concat: torch.Tensor,
        o_proj: Conv1D,
        n_iter: int = 5,
    ):
        """
        :param attn_weight_concat: concatenated attention weight matrices of
        shape (n_emb, 3 * n_emb)
        :param attn_bias_concat: concatenated attention bias of shape (3 * n_emb,)
        """

        decomposition_shapes = config.attn_decomposition_shapes

        hidden_size = config.hidden_size
        weight_q, weight_k, weight_v = [
            w for w in attn_weight_concat.split(hidden_size, dim=1)
        ]
        bias_q, bias_k, bias_v = [b for b in attn_bias_concat.split(hidden_size)]

        result = cls(config, layer_idx)

        (r, (m1, n1), (m2, n2)) = decomposition_shapes
        qproj = Conv1D(m1 * m2, n1 * n2)
        qproj.weight.data = weight_q.contiguous()
        qproj.bias.data = bias_q

        kproj = Conv1D(m1 * m2, n1 * n2)
        kproj.weight.data = weight_k.contiguous()
        kproj.bias.data = bias_k

        vproj = Conv1D(m1 * m2, n1 * n2)
        vproj.weight.data = weight_v.contiguous()
        vproj.bias.data = bias_v

        result.q_proj = layers.kn_pr_Conv1D.get_from_layer(
            qproj,
            r,
            m1,
            n1,
            m2,
            n2,
            verbose=True,
            n_iter_als=-1,
            n_iter=n_iter,
            method_type="random",
            use_p=True,
            use_q=True,
        )

        result.k_proj = layers.kn_pr_Conv1D.get_from_layer(
            kproj,
            r,
            m1,
            n1,
            m2,
            n2,
            verbose=True,
            n_iter_als=-1,
            n_iter=n_iter,
            method_type="random",
            use_p=True,
            use_q=True,
        )

        result.v_proj = layers.kn_pr_Conv1D.get_from_layer(
            vproj,
            r,
            m1,
            n1,
            m2,
            n2,
            verbose=True,
            n_iter_als=-1,
            n_iter=n_iter,
            method_type="random",
            use_p=True,
            use_q=True,
        )

        result.o_proj = o_proj

        return result
