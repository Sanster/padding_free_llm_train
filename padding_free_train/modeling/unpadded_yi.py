# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Unpadded & Fused LLaMA model. Compatible with HF. """

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from flash_attn.flash_attn_interface import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input

""" Yi model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

Yi_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class YiConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YiModel`]. It is used to instantiate an Yi
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Yi model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 64000):
            Vocabulary size of the Yi model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`YiModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048 or 4096).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to output attentions.
        rope_theta (`float`, *optional*, defaults to 5000000.0):
            The base period of the RoPE embeddings.
        Example:

    ```python
    >>> from transformers import YiModel, YiConfig

    >>> # Initializing a Yi style configuration
    >>> configuration = YiConfig()

    >>> # Initializing a model from the Yi style configuration
    >>> model = YiModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "Yi"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=64000,
        hidden_size=7168,
        intermediate_size=11008,
        num_hidden_layers=60,
        num_attention_heads=56,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        output_attentions=False,
        rope_theta=5000000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


logger = logging.get_logger(__name__)


@torch.jit.script  # type: ignore
def rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
):
    # q, k:     [nnz, num_heads, head_dim]
    # position_ids: [nnz]
    # cos, sin: [max_seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    sin = sin[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class UnpaddedYiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedYiRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class UnpaddedYiRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class UnpaddedYiMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnpaddedYiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: YiConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 56
        self.head_dim = self.hidden_size // self.num_heads  # 7168/ 56 = 128
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # nz_hidden_states: [nnz, num_heads, head_dim]
        # position_ids:  [nnz]
        # cu_seqlens:       [bs + 1]

        query_states = self.q_proj(nz_hidden_states).view(
            -1, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(nz_hidden_states).view(
            -1, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(nz_hidden_states).view(
            -1, self.num_key_value_heads, self.head_dim
        )

        # RoPE
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # flash attn
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            causal=True,
        )

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.hidden_size)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedYiDecoderLayer(nn.Module):
    def __init__(self, config: YiConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedYiAttention(config=config)
        self.mlp = UnpaddedYiMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.ln1 = UnpaddedYiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ln2 = UnpaddedYiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # Self Attention
        residual = nz_hidden_states

        nz_hidden_states = self.ln1(nz_hidden_states)
        nz_hidden_states = self.self_attn(
            cos_sin=cos_sin,
            nz_hidden_states=nz_hidden_states,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        nz_hidden_states = residual + nz_hidden_states

        # Fully Connected
        residual = nz_hidden_states

        nz_hidden_states = self.ln2(nz_hidden_states)
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states


class UnpaddedYiPreTrainedModel(PreTrainedModel):
    config_class = YiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedYiDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, UnpaddedYiModel):
    #         module.gradient_checkpointing = value


class UnpaddedYiModel(UnpaddedYiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedYiDecoderLayer`]

    Args:
        config: YiConfig
    """

    def __init__(self, config: YiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.rotary_emb = UnpaddedYiRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [UnpaddedYiDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = UnpaddedYiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        # Unpadded inputs
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        nz_hidden_states = self.embed_tokens(input_ids)
        cos_sin = self.rotary_emb()

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                nz_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    cos_sin,
                    nz_hidden_states,
                    position_ids,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                nz_hidden_states = decoder_layer(
                    cos_sin=cos_sin,
                    nz_hidden_states=nz_hidden_states,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

        nz_hidden_states = self.norm(nz_hidden_states)

        return nz_hidden_states


class YiForCausalLM(UnpaddedYiPreTrainedModel):
    # Ignore rotary emb inv_freq on load, as they will be calculated on creation
    _keys_to_ignore_on_load_unexpected = [
        r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedYiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        # Unpadded inputs
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        # Unpadded labels
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        # Model logits
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)


class PaddedYiForCausalLM(YiForCausalLM):
    """Compat layer for padded inputs"""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        # unused
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        # get indices
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )

        # Unpad inputs
        input_ids = torch.take_along_dim(input_ids, indices)
        position_ids = torch.take_along_dim(position_ids, indices)

        # Unpadded forward
        logits = (
            super()
            .forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_in_batch,
            )
            .logits
        )

        # Pad logits
        logits = pad_input(logits, indices, batch_size, seq_len)

        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "position_ids": kwargs.get("position_ids"),
        }


if __name__ == "__main__":
    model = YiForCausalLM.from_pretrained(
        "/mnt/.cache/huggingface/hub/models--01-ai--Yi-34B/snapshots/e2890c258236706e71284969e118abff5a536959"
    )
