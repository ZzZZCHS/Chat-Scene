import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.dataloader import MetaLoader
from dataset.dataset_train import TrainDataset
from dataset.dataset_val import ValDataset

import logging
logger = logging.getLogger(__name__)


def create_dataset(config):
    if config.evaluate:
        train_datasets = []
    else:
        train_files = []
        for train_name in config.train_tag.split('#'):
            if train_name not in config.train_file_dict:
                raise NotImplementedError
            train_files.append(config.train_file_dict[train_name])
        
        train_datasets = []
        datasets = []
        for train_file in train_files:
            datasets.append(TrainDataset(ann_list=train_file, config=config))
        dataset = ConcatDataset(datasets)
        train_datasets.append(dataset)

    val_files = {}
    for val_name in config.val_tag.split('#'):
        if val_name not in config.val_file_dict:
            raise NotImplementedError
        val_files[val_name] = config.val_file_dict[val_name]

    val_datasets = []
    for k, v in val_files.items():
        datasets = []
        if type(v[0]) != list:
            v = [v]
        for val_file in v:
            datasets.append(ValDataset(ann_list=val_file, dataset_name=k, config=config))
        dataset = ConcatDataset(datasets)
        val_datasets.append(dataset)

    return train_datasets, val_datasets


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders


def iterate_dataloaders(dataloaders):
    """Alternatively generate data from multiple dataloaders,
    since we use `zip` to concat multiple dataloaders,
    the loop will end when the smaller dataloader runs out.

    Args:
        dataloaders List(DataLoader): can be a single or multiple dataloaders
    """
    for data_tuples in zip(*dataloaders):
        for idx, data in enumerate(data_tuples):
            yield dataloaders[idx].dataset.media_type, data
