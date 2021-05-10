import torch
from torch.utils.data._utils.collate import default_collate

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler


class MultiTaskBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, drop_last, cum_thresholds):
        super().__init__(sampler, batch_size, drop_last)
        self.thresholds = cum_thresholds
        self.thres_ranges = list(zip(self.thresholds, self.thresholds[1:]))
        self.range_lens = [ed - st for st, ed in self.thres_ranges]

    def __iter__(self):
        batches = [[] for _ in self.thres_ranges]
        for idx in self.sampler:
            for range_idx, (st, ed) in enumerate(self.thres_ranges):
                if st <= idx < ed:
                    batches[range_idx].append(idx)
                    if len(batches[range_idx]) == self.batch_size:
                        yield batches[range_idx]
                        batches[range_idx] = []
        for range_idx in range(len(self.thres_ranges)):
            if len(batches[range_idx]) > 0 and not self.drop_last:
                yield batches[range_idx]

    def __len__(self):
        if self.drop_last:
            return sum([range_len // self.batch_size for range_len in self.range_lens])
        else:
            return sum([(range_len + self.batch_size - 1) // self.batch_size for range_len in self.range_lens])


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.
    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    Reference:
        torchnlp.samplers.distributed_batch_sampler
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)


def MultiTaskDataLoader(data_sources, shuffle, batch_size, drop_last, generator=None, **kwargs):
    dataset = ConcatDataset(data_sources)
    cum_thresholds = [0]
    for data_source in data_sources:
        cum_thresholds.append(cum_thresholds[-1] + len(data_source))
    if shuffle:
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = MultiTaskBatchSampler(sampler, batch_size=batch_size, drop_last=drop_last, cum_thresholds=cum_thresholds)
    batch_sampler = DistributedBatchSampler(batch_sampler)
    return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)

    
def online_mixing_collate(batch):
    """Mix target sources to create new mixtures.
    Output of the default collate function is expected to return two objects:
    inputs and targets.
    """
    # Inputs (batch, time) / targets (batch, n_src, time)
    inputs, targets = default_collate(batch)
    batch, n_src, _ = targets.shape

    energies = torch.sum(targets ** 2, dim=-1, keepdim=True)
    new_src = []
    for i in range(targets.shape[1]):
        new_s = targets[torch.randperm(batch), i, :]
        new_s = new_s * torch.sqrt(energies[:, i] / (new_s ** 2).sum(-1, keepdims=True))
        new_src.append(new_s)

    targets = torch.stack(new_src, dim=1)
    inputs = targets.sum(1)
    return inputs, targets
