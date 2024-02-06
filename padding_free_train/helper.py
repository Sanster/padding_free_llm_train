import math
import os
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from loguru import logger

from transformers import AutoModelForCausalLM, AutoTokenizer

from padding_free_train.multipack_sampler import MultipackDistributedBatchSampler


def save_args(model_args, data_args, train_args):
    output_dir = train_args.output_dir
    args_list = [model_args, data_args, train_args]
    if not args_list:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(
        os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8"
    ) as f:
        output = {}
        for args in args_list:
            output.update(asdict(args))
        json.dump(output, f, indent=2, ensure_ascii=False)


def get_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if "qwen" in model_name_or_path.lower():
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 151643
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_automodel_cls(model_name_or_path: str, padding_free: bool):
    from padding_free_train.modeling.unpadded_llama import LlamaForCausalLM
    from padding_free_train.modeling.unpadded_mistral import MistralForCausalLM
    from padding_free_train.modeling.unpadded_qwen import QWenLMHeadModel
    from padding_free_train.modeling.unpadded_qwen2 import Qwen2ForCausalLM

    if padding_free:
        if "llama" in model_name_or_path.lower():
            auto_model_cls = LlamaForCausalLM
        elif "qwen1.5" in model_name_or_path.lower():
            auto_model_cls = Qwen2ForCausalLM
        elif "qwen" in model_name_or_path.lower():
            auto_model_cls = QWenLMHeadModel
        elif "mistral" in model_name_or_path.lower():
            auto_model_cls = MistralForCausalLM
        else:
            raise NotImplementedError(f"Unknown model {model_name_or_path}")
    else:
        auto_model_cls = AutoModelForCausalLM
    return auto_model_cls


def calculate_auto_lr(batch_max_len, dataset, num_replicas):
    base_lr = 3e-4
    base_bs = 4_000_000
    labels = np.concatenate(dataset["labels"])
    supervised_ratio = np.sum(labels != -100) / len(labels)

    supervised_tokens = batch_max_len * num_replicas * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)

    print(
        f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})"
    )
    return lr


def convert_num_train_epochs_to_max_steps(training_args, dataset):
    total = 0
    tokens_lengths = [len(x["input_ids"]) for x in dataset]
    batch_sampler = MultipackDistributedBatchSampler(
        training_args.batch_max_length,
        tokens_lengths,
        seed=training_args.seed,
    )
    for epoch in range(math.ceil(training_args.num_train_epochs)):
        batch_sampler.set_epoch(epoch)
        epoch_steps = len(batch_sampler)
        logger.info(
            f"rank: {batch_sampler.rank} epoch: {epoch} epoch_steps: {epoch_steps}"
        )
        total += epoch_steps
    logger.info(
        f"Convert num_train_epochs [{training_args.num_train_epochs}] to max_steps: {total}"
    )
    training_args.max_steps = total
