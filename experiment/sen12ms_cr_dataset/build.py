from typing import Dict, Tuple, List
from sen12ms_cr_dataset.dataset import Roi, SEN12MSCRDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler


def build_distributed_loaders_with_rois(dataset_path: str, batch_size: int, file_extension: str, rois: List[Roi], debug: bool=False, return_with_triplet: bool=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SEN12MSCRDataset(dataset_path, file_extension, rois=rois, debug=debug, return_with_triplet=return_with_triplet)
    data_sampler = DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=data_sampler)


def build_loaders_with_rois(dataset_path: str, batch_size: int, file_extension: str, rois: List[Roi], debug: bool=False, return_with_triplet: bool=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SEN12MSCRDataset(dataset_path, file_extension, rois=rois, debug=debug, return_with_triplet=return_with_triplet)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


def build_loaders(dataset_path: str, batch_size: int, file_extension: str, debug: bool=False, return_with_triplet: bool=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Returns a tuple of dataloader, which are train_loader, val_loader, test_loader
    """
    dataset = SEN12MSCRDataset(dataset_path, file_extension, debug=debug, return_with_triplet=return_with_triplet)
    n_train = int(len(dataset) * 0.6)
    n_test = int(len(dataset) * 0.2)
    n_val = len(dataset) - n_train - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_distributed_loaders(dataset_path: str, batch_size: int, file_extension: str, debug: bool=False, return_with_triplet: bool=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SEN12MSCRDataset(dataset_path, file_extension, debug=debug, return_with_triplet=return_with_triplet)
    n_train = int(len(dataset) * 0.6)
    n_test = int(len(dataset) * 0.2)
    n_val = len(dataset) - n_train - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(val_set)
    test_sampler = DistributedSampler(test_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, sampler=test_sampler)
    return train_loader, val_loader, test_loader
