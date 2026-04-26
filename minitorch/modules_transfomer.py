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
from typing import Any, Dict, List, Optional, Sequence, Tuple

datatype = np.float32


def _to_numpy(storage) -> np.ndarray:
    if hasattr(storage, 'copy_to_host'):
        return storage.copy_to_host()
    return np.array(storage)


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        self.backend   = backend
        self.n_embd    = n_embd
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head

        self.q_projection   = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection   = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection   = Linear(n_embd, n_embd, bias=bias, backend=backend)
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

        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)

        kT = k.permute(0, 1, 3, 2)
        return q, kT, v

    def self_attention(self, q, kT, v, mask=None, past_kv=None):
        """
        Args:
            past_kv: tuple of (kT_np, v_np) numpy arrays from previous steps.
                     kT_np shape: (batch, n_head, attn_dim, past_seq)
                     v_np  shape: (batch, n_head, past_seq, attn_dim)
        Returns:
            output:  (batch, queries_len, n_embd)
            new_kv:  tuple of (kT_np, v_np) numpy arrays including current step
        """
        batch_size, num_head, queries_len, q_dim = q.shape

        # Extract numpy from current kT and v (.contiguous() ensures correct memory layout after permute)
        kT_np = _to_numpy(kT.contiguous()._tensor._storage).reshape(batch_size, num_head, q_dim, queries_len)
        v_np  = _to_numpy(v.contiguous()._tensor._storage).reshape(batch_size, num_head, queries_len, q_dim)

        # Concatenate with past cache if provided
        if past_kv is not None:
            past_kT_np, past_v_np = past_kv
            kT_np = np.concatenate([past_kT_np, kT_np], axis=-1)   # along seq dim
            v_np  = np.concatenate([past_v_np,  v_np],  axis=-2)   # along seq dim
            kT = tensor_from_numpy(kT_np, backend=self.backend)
            v  = tensor_from_numpy(v_np,  backend=self.backend)

        scores = q @ kT
        scale  = 1.0 / np.sqrt(self.attn_hidden_dim)
        scores = scores * scale

        if mask is not None:
            scores = scores + mask
        elif self.causal:
            full_seq = kT_np.shape[-1]
            causal_mask = -np.finfo(datatype).max * np.triu(
                np.ones((batch_size, num_head, queries_len, full_seq), dtype=datatype), 
                full_seq - queries_len + 1
            )
            scores = scores + tensor_from_numpy(causal_mask, backend=self.backend)

        attn_weights = softmax(scores, dim=3)
        attn_output  = attn_weights @ v

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output_flat = attn_output.view(batch_size * queries_len, self.n_embd)
        out = self.out_projection(attn_output_flat)

        return out.view(batch_size, queries_len, self.n_embd), (kT_np, v_np)

    def forward(self, x, mask=None, past_kv=None):
        """
        Returns:
            output: (batch, seq, n_embd)
            new_kv: (kT_np, v_np)
        """
        q, kT, v = self.project_to_query_key_value(x)
        return self.self_attention(q, kT, v, mask=mask, past_kv=past_kv)


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
        self.use_fused_kernel = use_fused_kernel

        if not self.use_fused_kernel:
            self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
            self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            self.ln_1_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_1_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))
            self.ln_2_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_2_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))

    def forward(self, x, mask=None, past_kv=None):
        """
        Args:
            past_kv: (kT_np, v_np) for this layer, or None
        Returns:
            x:      updated hidden states
            new_kv: (kT_np, v_np) for this layer
        """
        batch_size, seq_len, x_dim = x.shape

        if not self.use_fused_kernel:
            attn_out, new_kv = self.attention(self.ln_1(x), mask=mask, past_kv=past_kv)
            x = x + attn_out
            x = x + self.ff(self.ln_2(x))
        else:
            x_flat1 = x.view(batch_size * seq_len, x_dim)
            ln1_x = LayerNorm.apply(x_flat1, self.ln_1_gamma.value, self.ln_1_beta.value)
            ln1_x = ln1_x.view(batch_size, seq_len, x_dim)
            attn_out, new_kv = self.attention(ln1_x, mask=mask, past_kv=past_kv)
            x = x + attn_out

            x_flat2 = x.view(batch_size * seq_len, x_dim)
            ln2_x = LayerNorm.apply(x_flat2, self.ln_2_gamma.value, self.ln_2_beta.value)
            ln2_x = ln2_x.view(batch_size, seq_len, x_dim)
            x = x + self.ff(ln2_x)

        return x, new_kv


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
        self.backend          = backend
        self.n_embd           = n_embd
        self.n_vocab          = n_vocab
        self.n_layer          = n_layer
        self.use_fused_kernel = use_fused_kernel

        self.token_embeddings    = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)

        for i in range(1, n_layer + 1):
            setattr(self, f"t_layer_{i}", TransformerLayer(
                n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel,
                ffn_dim=4 * n_embd
            ))

        self.dropout = Dropout(p_dropout)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)

        if not self.use_fused_kernel:
            self.ln = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            self.ln_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))

    def forward(self, idx, mask=None, past_kvs=None, offset=0):
        """
        Args:
            idx:      (batch, seq_len) token ids
            mask:     optional 4D attention mask
            past_kvs: list of (kT_np, v_np) per layer, or None
            offset:   position offset when using KV-cache (number of cached tokens)
        Returns:
            logits:   (batch, seq_len, n_vocab)
            new_kvs:  list of (kT_np, v_np) per layer
        """
        batch_size, seq_len = idx.shape

        tok_emb = self.token_embeddings(idx)

        pos_indices = tensor_from_numpy(
            np.arange(offset, offset + seq_len, dtype=np.float32),
            backend=self.backend
        ).view(1, seq_len)
        pos_emb = self.position_embeddings(pos_indices)

        x = self.dropout(tok_emb + pos_emb)

        new_kvs = []
        for i in range(1, self.n_layer + 1):
            layer_past_kv = past_kvs[i - 1] if past_kvs is not None else None
            x, new_kv = getattr(self, f"t_layer_{i}")(x, mask=mask, past_kv=layer_past_kv)
            new_kvs.append(new_kv)

        if not self.use_fused_kernel:
            x = self.ln(x)
        else:
            x_flat = x.view(batch_size * seq_len, self.n_embd)
            x = LayerNorm.apply(x_flat, self.ln_gamma.value, self.ln_beta.value)
            x = x.view(batch_size, seq_len, self.n_embd)

        logits = self.lm_head(x.view(batch_size * seq_len, self.n_embd))
        return logits.view(batch_size, seq_len, self.n_vocab), new_kvs