import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.dataloader import MetaLoader
from dataset.dataset_stage1 import S1PTDataset
from dataset.dataset_stage2 import S2PTDataset
from dataset.dataset_stage3 import S3PTDataset
from dataset.dataset_val import ValPTDataset

import logging
logger = logging.getLogger(__name__)


def create_dataset(config):
    if config.model.stage == 1:
        config_train_file = config.train_file_s1
        config_val_file = config.val_file_s1
        train_dataset_cls = val_dataset_cls = S1PTDataset
    elif config.model.stage == 2:
        config_train_file = config.train_file_s2
        config_val_file = config.val_file_s2
        train_dataset_cls = S2PTDataset
        val_dataset_cls = ValPTDataset
    elif config.model.stage == 3:
        config_train_file = config.train_file_s3
        config_val_file = config.val_file_s3
        train_dataset_cls = S3PTDataset
        val_dataset_cls = ValPTDataset
    else:
        raise NotImplementedError

    logger.info(f"train_file: {config_train_file}")

    # convert to list of lists
    train_files = (
        [config_train_file] if isinstance(config_train_file[0], str) else config_train_file
    )
    val_files = (
        [config_val_file] if isinstance(config_val_file[0], str) else config_val_file
    )

    train_datasets = []
    datasets = []
    for train_file in train_files:
        dataset_kwargs = dict(
            ann_file=train_file,
            system_path=config.model.system_path,
            stage=config.model.stage
        )
        datasets.append(train_dataset_cls(**dataset_kwargs))
    dataset = ConcatDataset(datasets)
    train_datasets.append(dataset)

    val_datasets = []
    datasets = []
    for val_file in val_files:
        dataset_kwargs = dict(
            ann_file=val_file,
            system_path=config.model.system_path,
            stage=config.model.stage
        )
        datasets.append(val_dataset_cls(**dataset_kwargs))
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
