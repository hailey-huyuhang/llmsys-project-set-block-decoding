"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        
        # COPY FROM ASSIGN2_3
        weights_data = np.random.normal(0, 1, (num_embeddings, embedding_dim)) # random initialization
        self.weights = Parameter(tensor_from_numpy(weights_data, backend)) # if inputs are tensor/variable, will automatically requires_grad
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        
        # COPY FROM ASSIGN2_3
        # 1. bs: different sentences in a batch
        one_hot_x = one_hot(x, self.num_embeddings) # 3D, (bs, seq_len) -> (bs, seq_len, num_embeddings)
        flat_x = one_hot_x.view(bs * seq_len, self.num_embeddings) # 3D -> 2D
        out_flat = flat_x @ self.weights.value 
        return out_flat.view(bs, seq_len, self.embedding_dim)

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        # COPY FROM ASSIGN2_3
        if not self.training or self.p_dropout == 0:
            return x
        
        # retain probability: 1-self.p_dropout
        mask_data = np.random.binomial(1, 1 - self.p_dropout, size=x.shape) # only random once for each weight
        mask = tensor_from_numpy(mask_data, backend=x.backend)

        scale = 1.0 / (1.0 - self.p_dropout)
        return x * mask * scale


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.out_size = out_size
        
        # COPY FROM ASSIGN2_3
        limit = 1.0 / np.sqrt(in_size)

        # initialize weights, uniform Initialization
        weight_data = np.random.uniform(-limit, limit, (in_size, out_size))
        self.weights = Parameter(tensor_from_numpy(weight_data, backend=backend))

        # initialize bias if needed
        if bias:
            bias_data = np.random.uniform(-limit, limit, (out_size,))
            self.bias = Parameter(tensor_from_numpy(bias_data, backend=backend))
        else:
            self.bias = None

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        
        # COPY FROM ASSIGN2_3
        original_shape = x.shape
        
        import math
        batch_dim = 1
        for s in original_shape[:-1]:
            batch_dim *= s
            
        x_flat = x.view(batch_dim, original_shape[-1])
        
        out = x_flat @ self.weights.value
        
        if self.bias is not None:
            out = out + self.bias.value
            
        new_shape = original_shape[:-1] + (self.out_size,)
        return out.view(*new_shape)
    
        # batch, in_size = x.shape

        # out = x @ self.weights.value

        # if self.bias is not None:
        #     out = out + self.bias.value

        # return out.view(*prefix, self.out_size)


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        
        # COPY FROM ASSIGN2_3
        # all one weight
        self.weights = Parameter(ones_tensor_from_numpy((dim,), backend=backend))
        # all zero bias
        self.bias = Parameter(zeros_tensor_from_numpy((dim,), backend=backend))

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        # COPY FROM ASSIGN2_3
        original_shape = x.shape
        dim = original_shape[-1]
        
        batch_dim = 1
        for s in original_shape[:-1]:
            batch_dim *= s
            
        x_flat = x.contiguous().view(batch_dim, dim)
        
        mean = x_flat.mean(1).view(batch_dim, 1)
        
        var = ((x_flat - mean) ** 2).mean(1).view(batch_dim, 1)
        
        std = (var + self.eps) ** 0.5
        
        x_hat = (x_flat - mean) / std
        
        w = self.weights.value.view(1, dim)
        b = self.bias.value.view(1, dim)
        
        out = x_hat * w + b

        # back to original shape
        return out.view(*original_shape)
    
        # batch, dim = x.shape
        # ### BEGIN ASSIGN3_2
        # # along 1st dim(dim) get mean and var
        # mean = x.mean(1).view(batch, 1)
        # var = ((x - mean) ** 2).mean(1).view(batch, 1)

        # # denominator = (var + self.eps) ** 0.5
        # inv_std = (var + self.eps) ** (-0.5)
        # x_hat = (x - mean) * inv_std

        # # affine transform
        # return x_hat * self.weights.value + self.bias.value
        ### END ASSIGN3_2

