import torch
from torch.utils.data import Sampler
import math


class ResumableEpochSampler(Sampler):
    def __init__(self, dataset, seed, batch_size, num_processes):
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.epoch = 0
        self.resume_offset = 0

    def __iter__(self):
        if self.resume_offset > 0:
            yield from self._resume_iter()
        else:
            yield from self._normal_iter()

    def _normal_iter(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        # Drop last to make divisible by batch_size * num_processes
        total_size = (len(indices) // (self.batch_size * self.num_processes)) * self.batch_size * self.num_processes
        indices = indices[:total_size]

        for i in indices:
            yield i

    def _resume_iter(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        total_size = (len(indices) // (self.batch_size * self.num_processes)) * self.batch_size * self.num_processes
        indices = indices[:total_size]

        for i in indices[self.resume_offset :]:
            yield i
        self.resume_offset = 0

    def __len__(self):
        total = len(self.dataset)
        per_batch = self.batch_size * self.num_processes
        return total // per_batch * per_batch // self.num_processes

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_resume_offset(self, offset):
        self.resume_offset = offset
