# coding=utf-8
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import datasets
from transformers import PreTrainedTokenizer

SYSTEM_PREFIX = "<|im_start|>system\n"
QUERY_PREFIX = "<|im_start|>user\n"
ANSWER_PREFIX = "<|im_start|>assistant\n"
EOS_TOKEN = "<|im_end|>"

PREFIX_MAP = {
    "system": SYSTEM_PREFIX,
    "user": QUERY_PREFIX,
    "assistant": ANSWER_PREFIX,
}


def _tokenizer(tokenizer, text: str) -> List[int]:
    return tokenizer(text, return_attention_mask=False, add_special_tokens=False)[
        "input_ids"
    ]


def tokenize_conversation(
    tokenizer: PreTrainedTokenizer,
    conversation: List[Dict],
) -> Tuple[List[int], List[int]]:
    tokens = []
    labels = []

    last_idx = len(conversation) - 1
    for idx, msg in enumerate(conversation):
        if msg["role"] == "system" and msg["content"] == "":
            continue

        role_tokens = _tokenizer(tokenizer, PREFIX_MAP[msg["role"]])
        content_tokens = _tokenizer(tokenizer, msg["content"])
        im_end_tokens = _tokenizer(tokenizer, EOS_TOKEN)
        new_line_tokens = []
        if idx != last_idx:
            new_line_tokens = _tokenizer(tokenizer, "\n")

        if msg["role"] in ["system", "user"]:
            labels.extend(
                [-100] * (len(role_tokens) + len(content_tokens))
                + im_end_tokens
                + [-100] * len(new_line_tokens)
            )
        else:
            labels.extend(
                [-100] * len(role_tokens)
                + content_tokens
                + im_end_tokens
                + [-100] * len(new_line_tokens)
            )
        tokens.extend(role_tokens + content_tokens + im_end_tokens + new_line_tokens)

    assert len(tokens) == len(labels)
    return tokens, labels


@dataclass
class Config(datasets.BuilderConfig):
    jsonl: str = None
    tokenizer: PreTrainedTokenizer = None
    max_length: int = 4096


class ChatMLSFTDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = Config
    BUILDER_CONFIGS = [Config()]

    def _info(self):
        features = {
            "input_ids": datasets.Sequence(datasets.Value("int64")),
            "labels": datasets.Sequence(datasets.Value("int64")),
        }

        return datasets.DatasetInfo(
            description="SFT dataset",
            features=datasets.Features(features),
            supervised_keys=None,
            citation="SFT dataset",
        )

    def _split_generators(self, dl_manager):
        out = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "jsonl": self.config.jsonl,
                    "split": datasets.Split.TRAIN,
                },
            )
        ]

        return out

    def _generate_examples(self, jsonl: str, split):
        guid = 0
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                """
                {
                    {"role": "system", "content": "You are an AI assistant..."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "hi"},
                }
                """
                conversation = json.loads(line)
                input_ids, labels = tokenize_conversation(
                    tokenizer=self.config.tokenizer,
                    conversation=conversation,
                )

                yield guid, {
                    "input_ids": input_ids[: self.config.max_length],
                    "labels": labels[: self.config.max_length],
                }
                guid += 1
