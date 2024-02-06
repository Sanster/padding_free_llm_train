# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2PreTrainedModel,
    Qwen2MLP,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class UnpaddedQwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
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
        # qkv_states = self.c_attn(nz_hidden_states)
        # query, key, value = qkv_states.split(self.split_size, dim=2)
        query = self.q_proj(nz_hidden_states)
        key = self.k_proj(nz_hidden_states)
        value = self.v_proj(nz_hidden_states)

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
        return self.o_proj(attn_output)


class UnpaddedQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = UnpaddedQwen2Attention(config=config)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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

        nz_hidden_states = self.input_layernorm(nz_hidden_states)
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

        nz_hidden_states = self.post_attention_layernorm(nz_hidden_states)
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states


class UnpaddedQwen2RotaryEmbedding(torch.nn.Module):
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


class UnpaddedQwen2Model(Qwen2PreTrainedModel):
    """
    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = UnpaddedQwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [UnpaddedQwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
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


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    # Ignore rotary emb inv_freq on load, as they will be calculated on creation
    _keys_to_ignore_on_load_unexpected = [
        r"transformer\.h\.\d+\.self_attn\.rotary_emb\.inv_freq"
    ]

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = UnpaddedQwen2Model(config)
        self.vocab_size = config.vocab_size
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
        # print(
        #     f"input_ids: {input_ids.shape}, labels: {labels.shape}, position_ids: {position_ids.shape}, cu_seqlens: {cu_seqlens}, max_seqlen: {max_seqlen}"
        # )
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


class PaddedQWenForCausalLM(Qwen2ForCausalLM):
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
    from transformers import AutoTokenizer

    model_path = "/mnt/.cache/modelscope/hub/qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PaddedQWenForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")

    input_ids = tokenizer(["你是谁？我是由"], return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
