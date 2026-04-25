from .tensor_functions import Attn_Softmax, LayerNorm
import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head

        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)

        self.use_fused_kernel = use_fused_kernel

    def create_causal_mask(self, bs, nh, seq_len):
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        batch_size, seq_len, n_embd = x.shape
        
        x_flat = x.view(batch_size * seq_len, n_embd)
        
        q = self.q_projection(x_flat).view(batch_size, seq_len, self.n_embd)
        k = self.k_projection(x_flat).view(batch_size, seq_len, self.n_embd)
        v = self.v_projection(x_flat).view(batch_size, seq_len, self.n_embd)

        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        kT = k.permute(0, 1, 3, 2)
        
        return q, kT, v

    def self_attention(self, q, kT, v, mask=None):
        """
        Args:
            mask: optional 4D tensor (batch, heads, seq, seq) added to attention scores.
                  When None and self.causal is True, the standard causal mask is used.
                  When provided, it overrides the internal causal mask entirely.
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim

        scores = q @ kT
        scale = 1.0 / np.sqrt(self.attn_hidden_dim)
        scores = scores * scale

        if mask is not None:
            scores = scores + mask
        elif self.causal:
            causal_mask = self.create_causal_mask(batch_size, num_head, queries_len)
            scores = scores + causal_mask

        if not self.use_fused_kernel or scores.shape[-1] % 4 != 0:
            attn_weights = softmax(scores, dim=3)
            attn_output = attn_weights @ v
        else:
            out = Attn_Softmax.apply(scores, mask)
            attn_output = out @ v

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output_flat = attn_output.view(batch_size * queries_len, self.n_embd)
        out = self.out_projection(attn_output_flat)
        
        return out.view(batch_size, queries_len, self.n_embd)

    def forward(self, x, mask=None):
        """
        Args:
            mask: optional 4D attention mask passed through to self_attention.
        """
        q, kT, v = self.project_to_query_key_value(x)
        return self.self_attention(q, kT, v, mask=mask)


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        return x


class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False, ffn_dim=None):
        super().__init__()
        self.attention = MultiHeadAttention(
            n_embd, n_head, causal=True, p_dropout=p_dropout,
            bias=bias, backend=backend, use_fused_kernel=use_fused_kernel
        )
        self.ff = FeedForward(
            n_embd, 
            middle_dim=ffn_dim if ffn_dim else 4 * n_embd,
            p_dropout=p_dropout, bias=bias, backend=backend
        )
        # self.ff = FeedForward(
        #     n_embd, p_dropout=p_dropout, bias=bias, backend=backend
        # )
        self.use_fused_kernel = use_fused_kernel

        if not self.use_fused_kernel:
            self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
            self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            self.ln_1_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_1_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))
            self.ln_2_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_2_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))

    def forward(self, x, mask=None):
        """
        Args:
            mask: optional 4D attention mask passed through to MultiHeadAttention.
        """
        batch_size, seq_len, x_dim = x.shape

        if not self.use_fused_kernel:
            x = x + self.attention(self.ln_1(x), mask=mask)
            x = x + self.ff(self.ln_2(x))
        else:
            x_flat1 = x.view(batch_size * seq_len, x_dim)
            ln1_x = LayerNorm.apply(x_flat1, self.ln_1_gamma.value, self.ln_1_beta.value)
            ln1_x = ln1_x.view(batch_size, seq_len, x_dim)
            x = x + self.attention(ln1_x, mask=mask)

            x_flat2 = x.view(batch_size * seq_len, x_dim)
            ln2_x = LayerNorm.apply(x_flat2, self.ln_2_gamma.value, self.ln_2_beta.value)
            ln2_x = ln2_x.view(batch_size, seq_len, x_dim)
            x = x + self.ff(ln2_x)

        return x


class DecoderLM(Module):
    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int = 4,
        p_dropout: float=0.1,
        ln_eps: float=1e-5,
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
    ):
        super().__init__()
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        self.use_fused_kernel    = use_fused_kernel

        self.token_embeddings    = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)

        # self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        # self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        # self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        # self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        for i in range(1, n_layer + 1):
            setattr(self, f"t_layer_{i}", TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel, ffn_dim=4 * n_embd))

        self.dropout = Dropout(p_dropout)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        self.n_layer = n_layer

        if not self.use_fused_kernel:
            self.ln = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            self.ln_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))

    def forward(self, idx, mask=None):
        """
        Args:
            idx:  input token ids of shape (batch_size, seq_len)
            mask: optional 4D attention mask of shape (batch_size, n_head, seq_len, seq_len).
                  When provided, it is passed to every TransformerLayer and overrides the
                  default causal mask inside MultiHeadAttention.
        """
        batch_size, seq_len = idx.shape

        tok_emb = self.token_embeddings(idx)

        pos_indices = tensor_from_numpy(np.arange(seq_len), backend=self.backend).view(1, seq_len)
        pos_emb = self.position_embeddings(pos_indices)

        x = self.dropout(tok_emb + pos_emb)

        # x = self.t_layer_1(x, mask=mask)
        # x = self.t_layer_2(x, mask=mask)
        # x = self.t_layer_3(x, mask=mask)
        # x = self.t_layer_4(x, mask=mask)
        for i in range(1, self.n_layer + 1):
            x = getattr(self, f"t_layer_{i}")(x, mask=mask)

        if not self.use_fused_kernel:
            x = self.ln(x)
        else:
            x_flat = x.view(batch_size * seq_len, self.n_embd)
            x = LayerNorm.apply(x_flat, self.ln_gamma.value, self.ln_beta.value)
            x = x.view(batch_size, seq_len, self.n_embd)

        logits = self.lm_head(x.view(batch_size * seq_len, self.n_embd))
        return logits.view(batch_size, seq_len, self.n_vocab)