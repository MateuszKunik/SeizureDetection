import torch
from torch.utils.data import DataLoader

from .custom_dataset import CustomDataset


def create_dataloaders(data: dict, parameters: dict) -> tuple:
    train_dataloader = create_single_dataloader(
        data,  subset_type="train", parameters=parameters)
    
    valid_dataloader = create_single_dataloader(
        data, subset_type="valid", parameters=parameters)
    
    test_dataloader = create_single_dataloader(
        data, subset_type="test", parameters=parameters)
    
    return train_dataloader, valid_dataloader, test_dataloader


def create_single_dataloader(
        data: dict,
        subset_type: str,
        parameters: dict
    )  -> torch.utils.data.DataLoader:
    data_subset = get_data_subset(data, subset_type)
    # miejsce do zdefiniowania lub przekazania transformacji danych
    transform = None 

    if is_train_type(subset_type):
        # miejsce do zdefiniowania lub przekazania augmentacji danych
        augmentation = None
        dataset = CustomDataset(data_subset, transform, augmentation)
        shuffle = True

    else:
        dataset = CustomDataset(data_subset, transform)
        shuffle=False
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=parameters["batch_size"],
        num_workers=parameters["num_workers"],
        pin_memory=parameters["pin_memory"],
        shuffle=shuffle)
    
    return dataloader


def get_data_subset(data, subset_type):
    return data[subset_type]

def is_train_type(subset_type: str) -> bool:
    return subset_type == "train"