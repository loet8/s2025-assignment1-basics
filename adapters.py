#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import IO, BinaryIO, Iterable, Iterator, Optional, Type, Dict, List, Tuple

import numpy.typing as npt
import regex as re
import torch
from torch.optim import Optimizer
import math
from collections import Counter, defaultdict

epsilon = 1e-5
GPT2_TOKENIZER_REGEX = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are `w1.weight` and `w2.weight`.
            `w1` is the first linear transformation, and `w2` is the second
            linear transformation (eq. 2 of Vaswani et al., 2017).
            `w1.weight` is of shape (d_ff, d_model).
            `w2.weight` is of shape (d_model, d_ff).
    )
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your position-wise feedforward network
        with the provided `weights` on the provided `in_features`.        
    """

    assert "w1.weight" in weights, "Missing 'w1.weight' in weights"
    assert "w2.weight" in weights, "Missing 'w2.weight' in weights"

    first = torch.matmul(in_features, weights["w1.weight"].T)

    first = run_gelu(first)

    second = torch.matmul(first, weights["w2.weight"].T)

    return second.to(dtype=torch.float)
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # my_ffn.load_state_dict(weights)
    # You can also manually assign the weights
    # my_ffn.w1.weight.data = weights["w1.weight"]
    # my_ffn.w2.weight.data = weights["w2.weight"]
    raise NotImplementedError


def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

       
    d_k = math.sqrt(K.shape[-1])  
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, -1e9)  

    attn_probs = run_softmax(attn_scores, dim=-1)

    if pdrop is not None and pdrop > 0.0:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=pdrop, training=True)

    output = torch.matmul(attn_probs, V)

    return output.to(dtype=torch.float)

    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `q_heads.{N}.weight`, `q_heads.{N}.weight`:
                Weights for the query projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `k_heads.{N}.weight`, `k_heads.{N}.weight`:
                Weights for the key projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_key, d_model).
            - `v_heads.{N}.weight`, `v_heads.{N}.weight`:
                Weights for the value projection heads.
                N is an integer from 0 to `num_heads - 1`.
                Shape of each tensor is (d_value, d_model).
            - `output_proj.weight`:
                Weight of the output projection
                (W^{O} in the original Transformer paper)
                Shape of (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """


    batch_size, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads

    if "q_proj.weight" in weights:
        q_proj = weights["q_proj.weight"]  
        k_proj = weights["k_proj.weight"]
        v_proj = weights["v_proj.weight"]
        Q = torch.matmul(in_features, q_proj.T)
        K = torch.matmul(in_features, k_proj.T)
        V = torch.matmul(in_features, v_proj.T)
    else:
        Q = torch.cat([torch.matmul(in_features, weights[f"q_heads.{i}.weight"].T) for i in range(num_heads)], dim=-1)
        K = torch.cat([torch.matmul(in_features, weights[f"k_heads.{i}.weight"].T) for i in range(num_heads)], dim=-1)
        V = torch.cat([torch.matmul(in_features, weights[f"v_heads.{i}.weight"].T) for i in range(num_heads)], dim=-1)

    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    attn_probs = run_softmax(attn_scores, dim=-1)
    attn_probs = torch.nn.functional.dropout(attn_probs, p=attn_pdrop)
    attn_output = torch.matmul(attn_probs, V)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
    output_proj_weight = weights["output_proj.weight"]
    output = torch.matmul(attn_output, output_proj_weight.T)
    return output
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, (d_model / num_heads) * num_heads).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.
            Shape is (batch_size, sequence_length, d_model).

    Returns:
        FloatTensor of shape (batch_size, sequence_length, d_model) with the output of
        running the Transformer block on the input features.
    """

    x_norm1 = run_rmsnorm(d_model, epsilon, {"weight": weights["ln1.weight"]}, in_features)

    head_dim = d_model // num_heads
    if "attn.q_proj.weight" in weights:
        q_proj = weights["attn.q_proj.weight"].reshape(num_heads, head_dim, d_model)
        k_proj = weights["attn.k_proj.weight"].reshape(num_heads, head_dim, d_model)
        v_proj = weights["attn.v_proj.weight"].reshape(num_heads, head_dim, d_model)
        attn_weights = {f"q_heads.{i}.weight": q_proj[i] for i in range(num_heads)}
        attn_weights.update({f"k_heads.{i}.weight": k_proj[i] for i in range(num_heads)})
        attn_weights.update({f"v_heads.{i}.weight": v_proj[i] for i in range(num_heads)})
        attn_weights["output_proj.weight"] = weights["attn.output_proj.weight"]
    else:
        attn_weights = {f"q_heads.{i}.weight": weights[f"attn.q_heads.{i}.weight"] for i in range(num_heads)}
        attn_weights.update({f"k_heads.{i}.weight": weights[f"attn.k_heads.{i}.weight"] for i in range(num_heads)})
        attn_weights.update({f"v_heads.{i}.weight": weights[f"attn.v_heads.{i}.weight"] for i in range(num_heads)})
        attn_weights["output_proj.weight"] = weights["attn.output_proj.weight"]

    attn_output = run_multihead_self_attention(d_model, num_heads, attn_pdrop, attn_weights, x_norm1)
    attn_output = torch.nn.functional.dropout(attn_output, p=residual_pdrop, training=True)
    x_residual1 = in_features + attn_output

    x_norm2 = run_rmsnorm(d_model, epsilon, {"weight": weights["ln2.weight"]}, x_residual1)
    ffn_weights = {"w1.weight": weights["ffn.w1.weight"], "w2.weight": weights["ffn.w2.weight"]}
    ffn_output = run_positionwise_feedforward(d_model, d_ff, ffn_weights, x_norm2)
    ffn_output = torch.nn.functional.dropout(ffn_output, p=residual_pdrop, training=True)
    return x_residual1 + ffn_output

    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `position_embeddings.weight`
                Positional embedding matrix. Shape is (context_length, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices: torch.LongTensor
            Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
        next-word distribution for each token.
    """

    batch_size, sequence_length = in_indices.shape
    assert sequence_length <= context_length

    token_emb = weights["token_embeddings.weight"][in_indices]
    pos_emb = weights["position_embeddings.weight"][:sequence_length]
    x = token_emb + pos_emb
    x = torch.nn.functional.dropout(x, p=residual_pdrop, training=True)

    for layer in range(num_layers):
        if f"layers.{layer}.attn.q_proj.weight" in weights:
            layer_weights = {
                "ln1.weight": weights[f"layers.{layer}.ln1.weight"],
                "attn.q_proj.weight": weights[f"layers.{layer}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{layer}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{layer}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{layer}.attn.output_proj.weight"],
                "ln2.weight": weights[f"layers.{layer}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{layer}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{layer}.ffn.w2.weight"],
            }
        else:
            layer_weights = {
                "ln1.weight": weights[f"layers.{layer}.ln1.weight"],
                "attn.q_heads.0.weight": weights[f"layers.{layer}.attn.q_heads.0.weight"],
                "attn.q_heads.1.weight": weights[f"layers.{layer}.attn.q_heads.1.weight"],
                "attn.k_heads.0.weight": weights[f"layers.{layer}.attn.k_heads.0.weight"],
                "attn.k_heads.1.weight": weights[f"layers.{layer}.attn.k_heads.1.weight"],
                "attn.v_heads.0.weight": weights[f"layers.{layer}.attn.v_heads.0.weight"],
                "attn.v_heads.1.weight": weights[f"layers.{layer}.attn.v_heads.1.weight"],
                "attn.output_proj.weight": weights[f"layers.{layer}.attn.output_proj.weight"],
                "ln2.weight": weights[f"layers.{layer}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{layer}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{layer}.ffn.w2.weight"],
            }
        x = run_transformer_block(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, layer_weights, x)

    x = run_rmsnorm(d_model, epsilon, {"weight": weights["ln_final.weight"]}, x)
    logits = torch.matmul(x, weights["lm_head.weight"].T)
    return logits
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model: int
            The dimensionality of the RMSNorm input.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `weight`
                Weights of the RMSNorm affine transform.
                Shape is (d_model,).
        in_features: torch.FloatTensor
            Input features to run RMSNorm on. Tensor of (*, d_model), where *
            can be an arbitrary number of dimensions with arbitrary values.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """

    rms = torch.sqrt(torch.mean(in_features**2, dim=-1, keepdim=True) + eps)
    normalized = in_features / rms
    gamma = weights["weight"]  
    output = normalized * gamma
    return output

    raise NotImplementedError


def run_gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of applying GELU
    to each element.

    Args:
        in_features: torch.FloatTensor
            Input features to run GELU on. Shape is arbitrary.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of applying
        GELU to each element.
    """
    return in_features * 0.5 * (1 + torch.erf(in_features / math.sqrt(2.0)))

    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    if len(dataset) < context_length:
        raise ValueError("Dataset is too small for the specified context length.")
    
    max_start_idx = len(dataset) - context_length
    
    start_indices = torch.randint(0, max_start_idx, (batch_size,), dtype=torch.long)
    
    sequences: npt.NDArray = [dataset[i : i + context_length + 1] for i in start_indices]
    
    sequences = torch.tensor(sequences, dtype=torch.long, device=device)
    inputs = sequences[:, :-1]  
    targets = sequences[:, 1:]  
    
    return inputs, targets
    
    raise NotImplementedError


def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features: torch.FloatTensor
            Input features to softmax. Shape is arbitrary.
        dim: int
            Dimension of the `in_features` to apply softmax to.

    Returns:
        FloatTensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    max_vals = in_features.max(dim=dim, keepdim=True)[0]
    shifted_inputs = torch.subtract(in_features, max_vals)
    
    exp_values = torch.exp(shifted_inputs)
    
    sum_exp = exp_values.sum(dim=dim, keepdim=True)
    softmax_output = exp_values / (sum_exp + 1e-10)  
    
    return softmax_output


    raise NotImplementedError


def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    max_logits = inputs.max(dim=1, keepdim=True)[0]  

    exp_shifted = torch.exp(inputs - max_logits)  
    sum_exp = exp_shifted.sum(dim=1, keepdim=True)  
    log_probs = (inputs - max_logits) - torch.log(sum_exp)  

    loss = -log_probs[torch.arange(targets.shape[0]), targets]

    return loss.mean()
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    gradients = [p.grad for p in parameters if p.grad is not None]
    
    if not gradients:
        return  
    
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients) + epsilon)
    
    clip_coef = max_l2_norm / total_norm
    clip_coef = torch.clamp(clip_coef, max=1.0) 
    
    for g in gradients:
        g.mul_(clip_coef)

    #raise NotImplementedError

class AdamWOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Implements AdamW optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate (α).
            betas (Tuple[float, float]): Coefficients for first and second moment estimates (β1, β2).
            eps (float): Small value for numerical stability (ϵ).
            weight_decay (float): Weight decay coefficient (λ).
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()  
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                
                grad = param.grad.data

                if param not in self.state:
                    self.state[param] = {"step": 0, "m": torch.zeros_like(param), "v": torch.zeros_like(param)}

                state = self.state[param]
                state["step"] += 1
                step = state["step"]

                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad.pow(2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                adjusted_lr = lr * (bias_correction2 ** 0.5) / bias_correction1

                param.data -= adjusted_lr * state["m"] / (state["v"].sqrt() + eps)

                param.data -= lr * weight_decay * param.data

    def __str__(self):
        return "Custom AdamW Optimizer"

def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return torch.optim.AdamW

    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    else:
        return min_learning_rate
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model: torch.nn.Module
            Serialize the state of this model.
        optimizer: torch.optim.Optimizer,
            Serialize the state of this optimizer.
        iteration: int
            Serialize this value, which represents the number of training iterations
            we've completed.
        out: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),  
        "optimizer_state_dict": optimizer.state_dict(),  
        "iteration": iteration  
    }
    
    torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src: str | os.PathLike | BinaryIO | IO[bytes]
            Path or file-like object to serialized checkpoint.
        model: torch.nn.Module
            Restore the state of this model.
        optimizer: torch.optim.Optimizer,
            Restore the state of this optimizer.
    Returns:
        int, the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)  
    model.load_state_dict(checkpoint["model_state_dict"])  
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  
    
    return checkpoint["iteration"]


    raise NotImplementedError

class Tokenizer:
    def __init__(self, 
                 vocab: Dict[int, bytes], 
                 merges: List[Tuple[bytes, bytes]], 
                 special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        if len(set(vocab.values())) != len(vocab):
            raise ValueError("Vocab contains duplicate byte sequences!")

        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = set(special_tokens) if special_tokens else set()
        self.special_bytes = {s.encode("utf-8") for s in self.special_tokens}
        self.merge_dict = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, 
                   special_tokens: Optional[List[str]] = None):
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab = {int(k): bytes.fromhex(v) 
                     for k, v in (line.strip().split() for line in vf)}
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges = []
            for line in mf:
                cleaned_line = line.strip()
                if not cleaned_line or cleaned_line.startswith("#") or len(cleaned_line.split()) != 2:
                    continue
                m1, m2 = cleaned_line.split()
                merges.append((bytes.fromhex(m1), bytes.fromhex(m2)))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        

        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)

        tokenized_text = []
        i = 0
        while i < len(text):
            matched = False
            for token in sorted_specials:
                if text[i:].startswith(token):
                    tokenized_text.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                tokenized_text.append(text[i])
                i += 1


        tokens: List[bytes] = []
        for tok in tokenized_text:
            if tok.encode("utf-8") in self.special_bytes:
                tokens.append(tok.encode("utf-8"))
            else:
                tokens.extend([bytes([b]) for b in tok.encode("utf-8")])

        bpe_tokens = self._bpe_encode(tokens)

        token_ids = [self.byte_to_id.get(token, -1) for token in bpe_tokens]

        

        return token_ids


    def _bpe_encode(self, tokens: List[bytes]) -> List[bytes]:
        while len(tokens) > 1:
            pairs = list(zip(tokens, tokens[1:]))
            merge_candidates = []
            for pair in pairs:
                if pair == (b'\n', b'\n'):
                    merge_candidates.append((float("inf"), pair))
                else:
                    merge_candidates.append((self.merge_dict.get(pair, float("inf")), pair))
            min_rank = min(rank for rank, _ in merge_candidates)
            if min_rank == float("inf"):
                break  
            candidates = [pair for rank, pair in merge_candidates if rank == min_rank]
            best_pair = max(candidates)  
            new_tokens = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens


    def decode(self, ids: List[int]) -> str:
        return b"".join(self.vocab.get(i, b"?") for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
)-> Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes]
            The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges: list[tuple[bytes, bytes]]
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens: Optional[list[str]]
            A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """

    return Tokenizer(vocab, merges, special_tokens)

    raise NotImplementedError

def compute_pair_freqs(splits: Dict[bytes, List[bytes]], word_freqs: Dict[bytes, int]) -> Dict[Tuple[bytes, bytes], int]:
    """Compute the frequency of adjacent symbol pairs in tokenized words."""
    pair_freqs = defaultdict(int)

    for word, freq in word_freqs.items():
        split = splits[word]  
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq

    return pair_freqs


def merge_pair(a: bytes, b: bytes, splits: Dict[bytes, List[bytes]]) -> Dict[bytes, List[bytes]]:
    """Merge the most frequent pair in all tokenized words."""
    new_splits = {}

    for word, split in splits.items():
        new_split = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                new_split.append(a + b)  
                i += 2  
            else:
                new_split.append(split[i])
                i += 1
        new_splits[word] = new_split

    return new_splits

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
)-> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, text)
    word_freqs = Counter(word.encode("utf-8") for word in pre_tokens)

    vocab = {i: bytes([i]) for i in range(256)}
    token_id_counter = 256
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[token_id_counter] = token_bytes
            token_id_counter += 1

    splits = {word: [bytes([b]) for b in word] for word in word_freqs}
    merges: List[Tuple[bytes, bytes]] = []

    global_pair_freq = Counter()
    global_pair2words = defaultdict(set)
    for word, tokens in splits.items():
        if len(tokens) < 2:
            continue
        for pair in zip(tokens, tokens[1:]):
            global_pair_freq[pair] += word_freqs[word]
            global_pair2words[pair].add(word)

    while len(vocab) < vocab_size:
        if not global_pair_freq:
            break

        best_pair = max(global_pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        a, b = best_pair
        merged_token = a + b

        affected_words = list(global_pair2words[best_pair])
        for word in affected_words:
            old_tokens = splits[word]
            new_tokens = []
            i = 0
            n = len(old_tokens)
            while i < n:
                if i < n - 1 and old_tokens[i] == a and old_tokens[i+1] == b:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            splits[word] = new_tokens

            old_pairs = list(zip(old_tokens, old_tokens[1:]))
            new_pairs = list(zip(new_tokens, new_tokens[1:]))
            
            for pair in old_pairs:
                global_pair_freq[pair] -= word_freqs[word]
                if global_pair_freq[pair] <= 0:
                    del global_pair_freq[pair]
                    global_pair2words[pair].discard(word)
            
            for pair in new_pairs:
                global_pair_freq[pair] = global_pair_freq.get(pair, 0) + word_freqs[word]
                global_pair2words[pair].add(word)
        
        if best_pair in global_pair_freq:
            del global_pair_freq[best_pair]
        if best_pair in global_pair2words:
            del global_pair2words[best_pair]
        merges.append(best_pair)
        vocab[token_id_counter] = merged_token
        token_id_counter += 1

    return vocab, merges
    raise NotImplementedError
