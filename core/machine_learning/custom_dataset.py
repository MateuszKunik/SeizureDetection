import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class CustomDataset(Dataset):
    def __init__(
            self,
            data: tuple,
            transform: Compose = None,
            augmentation: Compose = None
    ):
        super().__init__()
        self.features = torch.from_numpy(data[0]).float()
        self.targets = torch.from_numpy(data[1]).long()
        self.transform = transform
        self.augmentation = augmentation


    def __len__(self):
        return self.features.shape[0]
    

    def __getitem__(self, index):
        features = self.features[index, :, :, :]
        target = self.targets[index]
        
        if self.transform:
            features = self.transform(features)

        if self.augmentation:
            features = self.augmentation(features)

        return features, target