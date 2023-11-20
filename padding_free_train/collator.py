from typing import List, Dict, Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def _find_multiple(a, b):
    return (-(a // -b)) * b


class ConcatDataCollator:
    def __init__(self, pad_token_id: int):
        assert pad_token_id is not None
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Concat batches
        batch_tensor = {}
        # fmt: off
        batch_tensor["position_ids"] = np.concatenate([np.arange(len(_["input_ids"])) for _ in batch], axis=0)
        batch_tensor["seqlens"] = np.array([len(_["input_ids"]) for _ in batch])
        batch_tensor["input_ids"] = np.concatenate([np.array(_["input_ids"]) for _ in batch], axis=0)
        batch_tensor["labels"] = np.concatenate([np.array(_["labels"]) for _ in batch], axis=0)
        # fmt: on

        # Pad an unused item to reach multiple of 64, for faster GEMM
        total_seqlen = batch_tensor["input_ids"].size
        pad_len = _find_multiple(total_seqlen, 64) - total_seqlen

        if pad_len > 0:
            assert pad_len < 64

            # total length
            padding_specs = {
                "seqlens": (1, pad_len),
                "input_ids": (pad_len, self.pad_token_id),
                "position_ids": (pad_len, 0),
                "labels": (pad_len, -100),
            }
            for k, pad_spec in padding_specs.items():
                batch_tensor[k] = np.concatenate(
                    (batch_tensor[k], np.full(*pad_spec, dtype=batch_tensor[k].dtype)),
                    axis=0,
                )

        for k in batch_tensor:
            batch_tensor[k] = torch.from_numpy(batch_tensor[k]).to(torch.long)
        batch_tensor["input_ids"] = batch_tensor["input_ids"].unsqueeze(0)

        # cu seqlens
        batch_tensor["cu_seqlens"] = torch.nn.functional.pad(
            batch_tensor["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0)
        )
        # batch info
        batch_tensor["max_seqlen"] = torch.max(batch_tensor["seqlens"])
        # inputs
        del batch_tensor["seqlens"]
        return batch_tensor


def left_pad_sequence(data: List[torch.Tensor], padding_value):
    return pad_sequence(
        [it.squeeze(0).flip(0) for it in data],
        batch_first=True,
        padding_value=padding_value,
    ).flip(1)


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]):
        # fmt: off
        batched_input_ids = left_pad_sequence([torch.LongTensor(it["input_ids"]) for it in features], self.pad_token_id)
        batched_labels = left_pad_sequence([torch.LongTensor(it["labels"]) for it in features], -100)
        # fmt: on

        return {
            "input_ids": batched_input_ids,
            "labels": batched_labels,
        }
