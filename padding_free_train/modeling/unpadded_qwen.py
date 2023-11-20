# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PretrainedConfig
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from flash_attn.flash_attn_interface import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input


logger = logging.get_logger(__name__)


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        emb_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_epsilon=1e-6,
        initializer_range=0.02,
        max_position_embeddings=4096,
        scale_attn_weights=True,
        use_cache=True,
        bf16=False,
        fp16=False,
        fp32=False,
        kv_channels=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        use_dynamic_ntk=True,
        use_logn_attn=True,
        use_flash_attn="auto",
        intermediate_size=22016,
        no_bias=True,
        tie_word_embeddings=False,
        shift_attention=False,
        shift_attention_group_size=8192,
        shift_attention_group_size_ratio=1 / 4,
        neftune=False,
        neftune_alpha=5,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.emb_dropout_prob = emb_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.bf16 = bf16
        self.fp16 = fp16
        self.fp32 = fp32
        self.kv_channels = kv_channels
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn
        self.no_bias = no_bias
        self.shift_attention = shift_attention
        self.shift_attention_group_size = shift_attention_group_size
        self.shift_attention_group_size_ratio = shift_attention_group_size_ratio
        self.neftune = neftune
        self.neftune_alpha = neftune_alpha
        """
        "rope_scaling": {
           "factor": 8.0,
           "type": "linear"/"static"
        }
        """
        self.rope_scaling = rope_scaling if rope_scaling is not None else {}
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


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


class UnpaddedQWenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class UnpaddedQWenRotaryEmbedding(torch.nn.Module):
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


class UnpaddedQWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        self.w2 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class UnpaddedQWenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: QWenConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.projection_size = config.kv_channels * config.num_attention_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)
        self.c_proj = nn.Linear(
            config.hidden_size, self.projection_size, bias=not config.no_bias
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
        qkv_states = self.c_attn(nz_hidden_states)
        query, key, value = qkv_states.split(self.split_size, dim=2)
        query_states = query.view(-1, self.num_heads, self.head_dim)
        key_states = key.view(-1, self.num_heads, self.head_dim)
        value_states = value.view(-1, self.num_heads, self.head_dim)

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
        return self.c_proj(attn_output)


class UnpaddedQWenDecoderLayer(nn.Module):
    def __init__(self, config: QWenConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.ln_1 = UnpaddedQWenRMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.attn = UnpaddedQWenAttention(config=config)
        self.ln_2 = UnpaddedQWenRMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = UnpaddedQWenMLP(config)

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

        nz_hidden_states = self.ln_1(nz_hidden_states)
        nz_hidden_states = self.attn(
            cos_sin=cos_sin,
            nz_hidden_states=nz_hidden_states,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        nz_hidden_states = residual + nz_hidden_states

        # Fully Connected
        residual = nz_hidden_states

        nz_hidden_states = self.ln_2(nz_hidden_states)
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states


class UnpaddedQWenPreTrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedQWenDecoderLayer"]

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, UnpaddedQWenRMSNorm):
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.num_hidden_layers)
                    ),
                )

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, UnpaddedQWenModel):
    #         module.gradient_checkpointing = value


class UnpaddedQWenModel(UnpaddedQWenPreTrainedModel):
    """
    Args:
        config: QWenConfig
    """

    def __init__(self, config: QWenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = UnpaddedQWenRotaryEmbedding(
            dim=config.kv_channels,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rotary_emb_base,
        )

        self.h = nn.ModuleList(
            [UnpaddedQWenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = UnpaddedQWenRMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self,
        # Unpadded inputs
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        nz_hidden_states = self.wte(input_ids)
        cos_sin = self.rotary_emb()

        # decoder layers
        for decoder_layer in self.h:
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

        nz_hidden_states = self.ln_f(nz_hidden_states)

        return nz_hidden_states


class QWenLMHeadModel(UnpaddedQWenPreTrainedModel):
    # Ignore rotary emb inv_freq on load, as they will be calculated on creation
    _keys_to_ignore_on_load_unexpected = [
        r"transformer\.h\.\d+\.self_attn\.rotary_emb\.inv_freq"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = UnpaddedQWenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

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
        # print(
        #     f"input_ids: {input_ids.shape}, labels: {labels.shape}, position_ids: {position_ids.shape}, cu_seqlens: {cu_seqlens}, max_seqlen: {max_seqlen}"
        # )
        hidden_states = self.transformer(
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


class PaddedQWenForCausalLM(QWenLMHeadModel):
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
    model = QWenForCausalLM.from_pretrained(
        "/mnt/.cache/huggingface/hub/models--Qwen--Qwen-7B-Chat/snapshots/013d71a2b7be6824faf45c30534fbea570970468/"
    )
