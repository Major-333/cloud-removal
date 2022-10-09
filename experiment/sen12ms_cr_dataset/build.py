from typing import Dict, Tuple
from sen12ms_cr_dataset.dataset import SEN12MSCRDataset
from torch.utils.data import DataLoader, random_split


# TODO: Dataset Split By CSV
def build_loaders(dataset_path: str, batch_size: int, file_extension: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ Returns a tuple of dataloader, which are train_loader, val_loader, test_loader
    """
    dataset = SEN12MSCRDataset(dataset_path, file_extension)
    n_train = int(len(dataset) * 0.6)
    n_test = int(len(dataset) * 0.2)
    n_val = len(dataset) - n_train - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    return train_loader, val_loader, test_loader
