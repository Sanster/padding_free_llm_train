#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
from typing import Optional, List

from padding_free_train.collator import (
    ConcatDataCollator,
    DataCollator,
)
from padding_free_train.helper import (
    save_args,
    get_automodel_cls,
    get_tokenizer,
    convert_num_train_epochs_to_max_steps,
)
from padding_free_train.my_trainer import MyTrainer

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from dataclasses import dataclass, field

import datasets
import transformers
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_FILE = os.path.join(CURRENT_DIR, "padding_free_train", "dataset.py")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    jsonl: Optional[str] = field(default=None, metadata={"help": "Training data"})


@dataclass
class MyTrainingArguments(TrainingArguments):
    sample_max_length: int = field(
        default=4096, metadata={"help": "Maximum length limit for a single sample."}
    )
    batch_max_length: int = field(
        default=4096 * 8,
        metadata={"help": "Maximum allowed length after padding for multiple samples."},
    )
    padding_free: bool = field(
        default=True,
        metadata={"help": "Enable padding-free + MultiPackDataloader"},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    weight_decay: float = field(
        default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.95, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-5, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    report_to: Optional[List[str]] = field(
        default="tensorboard",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )

    def __post_init__(self):
        if self.padding_free:
            if self.per_device_train_batch_size != 1:
                logger.warning(
                    f"per_device_train_batch_size does not work for padding-free training and has been set to 1."
                    f"If you want to increase the batch size, you can increase the batch_max_length."
                )
                self.per_device_train_batch_size = 1

        if self.sample_max_length > self.batch_max_length:
            raise ValueError(
                f"sample_max_length ({self.sample_max_length}) cannot be greater than batch_max_length ({self.batch_max_length})."
            )

        super().__post_init__()


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    model_name_or_path = model_args.model_name_or_path
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    tokenizer = get_tokenizer(model_name_or_path)
    dataset = load_dataset(
        DATASET_FILE,
        jsonl=data_args.jsonl,
        tokenizer=tokenizer,
        max_length=training_args.sample_max_length,
    )["train"]

    if training_args.padding_free:
        convert_num_train_epochs_to_max_steps(training_args, dataset)

    auto_model_cls = get_automodel_cls(model_name_or_path, training_args.padding_free)
    model = auto_model_cls.from_pretrained(
        model_name_or_path,
        use_cache=False,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()  # reduce number of stored activations

    trainer_kwargs = dict(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )

    if training_args.padding_free:
        trainer = MyTrainer(
            **trainer_kwargs,
            data_collator=ConcatDataCollator(pad_token_id=tokenizer.pad_token_id),
        )
    else:
        trainer = Trainer(
            **trainer_kwargs,
            data_collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
        )
    trainer.multi_pack_sampler_batch_max_length = training_args.batch_max_length
    trainer.accelerator.dispatch_batches = False

    if trainer.state.is_world_process_zero:
        save_args(
            model_args,
            data_args,
            training_args,
        )

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
