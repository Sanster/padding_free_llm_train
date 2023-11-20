from typing import Optional

import datasets
import torch
from torch.utils.data import RandomSampler, DataLoader
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import has_length, seed_worker

from padding_free_train.multipack_sampler import MultipackDistributedBatchSampler


class MyDataLoader(DataLoader):
    def set_epoch(self, epoch: int):
        # This function is very important, ensuring that the data for each epoch is shuffled.
        # https://github.com/huggingface/transformers/pull/26850
        self.batch_sampler.set_epoch(epoch)


class MyTrainer(Trainer):
    multi_pack_sampler_batch_max_length: int = 32768

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        assert self.train_dataset is not None and has_length(self.train_dataset)
        tokens_lengths = [len(x["input_ids"]) for x in self.train_dataset]
        return MultipackDistributedBatchSampler(
            batch_max_length=self.multi_pack_sampler_batch_max_length,
            lengths=tokens_lengths,
            seed=self.args.seed,
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["batch_sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        self.accelerator.even_batches = False
        dataloader = MyDataLoader(train_dataset, **dataloader_params)
        return dataloader
