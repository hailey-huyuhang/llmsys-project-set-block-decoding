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
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head

        # COPY FROM ASSIGN2_4
        # initialization of q k v, out, dropout
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend) # all heads together
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)

        self.use_fused_kernel = use_fused_kernel

    def create_causal_mask(self, bs, nh, seq_len):
        """
        return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
        """
        # mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1) # This should be ok, but may be problematic -> the loss will be NaN in Assignment 3 because the mask will not broadcast correctly in the kernel.
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1) # Correct version for Assignment 3.
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention
        
        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # COPY FROM ASSIGN2_4
        # linear projection
        x_flat = x.view(batch_size * seq_len, n_embd)
        
        q = self.q_projection(x_flat).view(batch_size, seq_len, self.n_embd)
        k = self.k_projection(x_flat).view(batch_size, seq_len, self.n_embd)
        v = self.v_projection(x_flat).view(batch_size, seq_len, self.n_embd)

        # slice to multiple head
        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)

        q = q.permute(0, 2, 1, 3) # move head dim to the front to calculate in parallel
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        kT = k.permute(0, 1, 3, 2)
        
        return q, kT, v

    def self_attention(self, q, kT, v):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p), 
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None

        # Attention Scores: Q @ K.T
        scores = q @ kT

        # scale and softmax to attention weights
        scale = 1.0 / np.sqrt(self.attn_hidden_dim)
        scores = scores * scale

        # if using causal mask
        if self.causal:
            mask = self.create_causal_mask(batch_size, num_head, queries_len)
            scores = scores + mask
        
        if not self.use_fused_kernel or scores.shape[-1] % 4 != 0:
            # COPY FROM ASSIGN2_4
            attn_weights = softmax(scores, dim=3)

            # weights @ V
            attn_output = attn_weights @ v

        else:
            # BEGIN ASSIGN3_3
            out = Attn_Softmax.apply(scores, mask)
            attn_output = out @ v
            # END ASSIGN3_3

        # convert to original dim order
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()

        attn_output_flat = attn_output.view(batch_size * queries_len, self.n_embd)
        out = self.out_projection(attn_output_flat)
        
        return out.view(batch_size, queries_len, self.n_embd)

    def forward(self, x):
        """Computes MultiHeadAttention with causal masking if needed. 

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        # COPY FROM ASSIGN2_4
        q, kT, v = self.project_to_query_key_value(x)
        return self.self_attention(q, kT, v)


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.
        
        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers
        
        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        # COPY FROM ASSIGN2_4
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        # COPY FROM ASSIGN2_4
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)

        return x

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers
        
        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """
        
        # COPY FROM ASSIGN2_4
        self.attention = MultiHeadAttention(
            n_embd, n_head, causal=True, p_dropout=p_dropout, 
            bias=bias, backend=backend, use_fused_kernel=use_fused_kernel
        )
        self.ff = FeedForward(
            n_embd, p_dropout=p_dropout, bias=bias, backend=backend
        )

        self.use_fused_kernel = use_fused_kernel
        
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
            self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            # BEGIN ASSIGN3_3
            self.ln_1_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_1_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))
            
            self.ln_2_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_2_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))
            # END ASSIGN3_3

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape
        
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # using preLN: x + Layer(Norm(x))
            # attention block
            x = x + self.attention(self.ln_1(x))
            # FF block
            x = x + self.ff(self.ln_2(x))
        else:
            # BEGIN ASSIGN3_3
            # flat to 2D
            x_flat1 = x.view(batch_size * seq_len, x_dim)
            ln1_x = LayerNorm.apply(x_flat1, self.ln_1_gamma.value, self.ln_1_beta.value)
            # reverse to 3D
            ln1_x = ln1_x.view(batch_size, seq_len, x_dim)

            x = x + self.attention(ln1_x)
            
            # flat to 2D
            x_flat2 = x.view(batch_size * seq_len, x_dim)
            ln2_x = LayerNorm.apply(x_flat2, self.ln_2_gamma.value, self.ln_2_beta.value)
            # reverse to 3D
            ln2_x = ln2_x.view(batch_size, seq_len, x_dim)

            x = x + self.ff(ln2_x)
            # END ASSIGN3_3

        return x


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.
        
        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        
        # COPY FROM ASSIGN2_4
        # self.token_embeddings    = 
        # self.position_embeddings = 
        # self.t_layer_1           = 
        # self.t_layer_2           = 
        # self.t_layer_3           = 
        # self.t_layer_4           = 
        # self.dropout             = 
        # self.lm_head             = 
        self.use_fused_kernel = use_fused_kernel
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)

        # transformer layers
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel)

        # final layers
        self.dropout = Dropout(p_dropout)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        ### END ASSIGN3_3
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # self.ln                  = 
            self.ln = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            # BEGIN ASSIGN3_3
            self.ln_gamma = Parameter(tensor_from_numpy(np.ones(n_embd, dtype=datatype), backend=backend))
            self.ln_beta  = Parameter(tensor_from_numpy(np.zeros(n_embd, dtype=datatype), backend=backend))
            # END ASSIGN3_3
        
    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args: 
            idx: input of shape (batch_size, seq_len)
        
        Returns: 
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape
        # pos = tensor([i for i in range(seq_len)], backend=self.backend).view(1, seq_len)

        # if not self.use_fused_kernel:
        #     # COPY FROM ASSIGN2_4
        #     raise NotImplementedError
        # else:
        #     # BEGIN ASSIGN3_3
        #     raise NotImplementedError
        #     # END ASSIGN3_3
        ### BEGIN ASSIGN3_3

        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        tok_emb = self.token_embeddings(idx) # (Batch, Seq, Dim)

        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        #    - Pass through positional embedding layer
        #    - Ensure output shape is (1, seq_len, n_embd)
        pos_indices_data = np.arange(seq_len) # assign index for each token
        pos_indices = tensor_from_numpy(pos_indices_data, backend=self.backend)
        pos_indices = pos_indices.view(1, seq_len) # change to row

        pos_emb = self.position_embeddings(pos_indices) # (1,     Seq, Dim)
        # 3. Add token and positional embeddings
        x = tok_emb + pos_emb # broadcast

        # 4. Apply dropout
        x = self.dropout(x)

        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)

        # 6. Apply final layer normalization
        if not self.use_fused_kernel:
            x = self.ln(x)
        else:
            # flatten to 2D
            x_flat = x.view(batch_size * seq_len, self.n_embd)
            x = LayerNorm.apply(x_flat, self.ln_gamma.value, self.ln_beta.value)
            # revert to 3D
            x = x.view(batch_size, seq_len, self.n_embd)

        # 7. Project to vocabulary size using lm_head
        x_flat = x.view(batch_size*seq_len, self.n_embd)
        logits = self.lm_head(x_flat)
        
        return logits.view(batch_size, seq_len, self.n_vocab)

        # return x
