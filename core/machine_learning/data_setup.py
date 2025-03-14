import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .custom_dataset import CustomDataset
from .custom_augmentation import RandomChannelShuffle


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
    transform = None 

    if is_train_type(subset_type):
        augmentation = RandomChannelShuffle(parameters["shuffle_prob"])
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