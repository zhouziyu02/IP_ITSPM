# Modified: replace external documentation links with local descriptions.
import math
from typing import Dict

import numpy as np
from torch.utils.data import Dataset, DataLoader

from easy_tpp.preprocess.data_collator import TPPDataCollator
from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import py_assert


class TPPDataset(Dataset):
    def __init__(self, data: Dict):
        self.data_dict = data
        self.time_seqs = self.data_dict['time_seqs']
        self.time_delta_seqs = self.data_dict['time_delta_seqs']
        self.type_seqs = self.data_dict['type_seqs']
        lengths = [len(self.time_seqs), len(self.time_delta_seqs), len(self.type_seqs)]
        if len(set(lengths)) != 1:
            raise ValueError("Time, interval, and event-type collections must have the same length.")
        for index, (times, deltas, marks) in enumerate(zip(
                self.time_seqs, self.time_delta_seqs, self.type_seqs)):
            if not len(times) or len(times) != len(deltas) or len(times) != len(marks):
                raise ValueError(f"Sequence {index} has empty or misaligned event fields.")
            if not np.isfinite(times).all() or not np.isfinite(deltas).all():
                raise ValueError(f"Sequence {index} contains non-finite timestamps or intervals.")
            if np.any(np.diff(times) < 0) or np.any(np.asarray(deltas) < 0):
                raise ValueError(f"Sequence {index} has decreasing timestamps or negative intervals.")

    def __len__(self):
        """

        Returns: length of the dataset

        """

        py_assert(len(self.time_seqs) == len(self.type_seqs) and len(self.time_delta_seqs) == len(self.type_seqs),
                  ValueError,
                  f"Inconsistent lengths for data! time_seq_len:{len(self.time_seqs)}, event_len: "
                  f"{len(self.type_seqs)}, time_delta_seq_len: {len(self.time_delta_seqs)}")

        return len(self.time_seqs)

    def __getitem__(self, idx):
        """

        Args:
            idx: iteration index

        Returns:
            dict: a dict of time_seqs, time_delta_seqs and type_seqs element

        """
        return dict({'time_seqs': self.time_seqs[idx], 'time_delta_seqs': self.time_delta_seqs[idx],
                     'type_seqs': self.type_seqs[idx]})

    def get_dt_stats(self):
        """Return population statistics of observed intervals, excluding the first event."""
        mean, squared_deviations, count = 0., 0., 0
        min_dt, max_dt = np.inf, -np.inf

        for dts, marks in zip(self.time_delta_seqs, self.type_seqs):
            dts = np.asarray(dts[1:-1 if marks[-1] == -1 else None], dtype=np.float64)
            if not dts.size:
                continue
            min_dt = min(min_dt, dts.min())
            max_dt = max(max_dt, dts.max())
            batch_count = dts.size
            batch_mean = dts.mean()
            mean_delta = batch_mean - mean
            total_count = count + batch_count
            squared_deviations += ((dts - batch_mean) ** 2).sum()
            squared_deviations += mean_delta ** 2 * count * batch_count / total_count
            mean += mean_delta * batch_count / total_count
            count = total_count
        if not count:
            raise ValueError("At least one observed inter-event interval is required.")
        return mean, math.sqrt(max(squared_deviations / count, 0.)), min_dt, max_dt


def get_data_loader(dataset: TPPDataset, backend: str, tokenizer: EventTokenizer, **kwargs):
    assert backend == 'torch', 'Only torch backend is supported.'
    padding = True if tokenizer.padding_strategy is None else tokenizer.padding_strategy
    truncation = False if tokenizer.truncation_strategy is None else tokenizer.truncation_strategy
    data_collator = TPPDataCollator(tokenizer=tokenizer,
                                    return_tensors='pt',
                                    max_length=tokenizer.model_max_length,
                                    padding=padding,
                                    truncation=truncation)
    return DataLoader(dataset,
                      collate_fn=data_collator,
                      **kwargs)
