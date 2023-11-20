import math
import os
from datasets import load_dataset
from pathlib import Path
import numpy as np

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from padding_free_train.collator import ConcatDataCollator, ConcatDataCollator2
from padding_free_train.helper import get_tokenizer, calculate_auto_lr
from padding_free_train.multipack_sampler import MultipackDistributedBatchSampler

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DATASET_FILE = str(CURRENT_DIR.parent / "padding_free_train" / "dataset.py")
DATA_FILE = str(
    CURRENT_DIR.parent / "data" / "random_selected_sharegpt_gpt4_en_zh.jsonl"
)


def test_load_dataset():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B-Chat", trust_remote_code=True
    )
    dataset = load_dataset(
        DATASET_FILE, jsonl=DATA_FILE, tokenizer=tokenizer, max_length=4096
    )["train"]
    assert len(dataset) == 1000
    token_lengths = [len(x["input_ids"]) for x in dataset]
    print(len(token_lengths))
    assert len(token_lengths) == 1000


def test_multi_pack_sampler():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B-Chat", trust_remote_code=True
    )
    dataset = load_dataset(DATASET_FILE, jsonl=DATA_FILE, tokenizer=tokenizer)["train"]
    lengths = [len(x["input_ids"]) for x in dataset]
    print(lengths)
    batch_max_length = 4096 * 2

    num_replicas_list = [1, 2, 8]
    for num_replicas in num_replicas_list:
        print(f"===========test num_replicas: {num_replicas}===========")
        all_indexes = set(list(range(len(dataset))))

        for rank in range(num_replicas):
            sampler = MultipackDistributedBatchSampler(
                batch_max_length=batch_max_length,
                lengths=lengths,
                num_replicas=num_replicas,
                rank=rank,
            )
            for batch_i, indexes in enumerate(sampler):
                print(f"rank: {rank}, [{batch_i}] indexes: {indexes}")
                for i in indexes:
                    assert i in all_indexes
                    all_indexes.remove(i)
            print(f"efficiency: {sampler.efficiency()}")


def test_dataloader():
    tokenizer = get_tokenizer("Qwen/Qwen-7B-Chat")
    dataset = load_dataset(
        DATASET_FILE, jsonl=DATA_FILE, tokenizer=tokenizer, max_length=4096
    )["train"]
    lengths = [len(x["input_ids"]) for x in dataset]
    sampler = MultipackDistributedBatchSampler(
        batch_max_length=8192,
        lengths=lengths,
        num_replicas=1,
        rank=0,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=ConcatDataCollator2(pad_token_id=tokenizer.pad_token_id),
    )

    for it in loader:
        print(it)


def test_auto_lr():
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B-Chat", trust_remote_code=True
    )
    batch_max_len = 8192
    dataset = load_dataset(
        DATASET_FILE, jsonl=DATA_FILE, tokenizer=tokenizer, max_length=batch_max_len
    )["train"]

    lr = calculate_auto_lr(batch_max_len, dataset, 4)
    print(lr)
