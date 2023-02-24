
import os

import pytorch_lightning as pl
# from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
from torch.utils.data import TensorDataset


import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from torchvision import transforms, datasets


class CIFARDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        image_size = (32, 32)

        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.247, 0.243, 0.261)

        self.train_transform = transforms.Compose([
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
            transforms.RandomResizedCrop(32)
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(data_mean, data_std),
        ])

    def setup(self, stage=None):

        self.dataset_train = datasets.CIFAR10(self.args.data_path, train=True,
                                              transform=self.train_transform, download=False)

        self.dataset_val = datasets.CIFAR10(self.args.data_path, train=False,
                                            transform=self.test_transform, download=False)

        self.dataset_test = {d: CifarCorrupted(self.args.base_c_path, d, self.args.only_severe_corruption,
                                               transform=self.test_transform)
                             for d in self.args.corruptions}

        return

    def train_dataloader(self) -> DataLoader:
        # choose the training and test datasets
        train_loader = DataLoader(self.dataset_train,
                                  batch_size=self.args.batch_size,
                                  num_workers=8,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        return train_loader

    def val_dataloader(self):
        val_loaders = [DataLoader(self.dataset_test[name],
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=5,
                                  pin_memory=True, )
                       for name in self.dataset_test.keys()]

        val_loaders.append(DataLoader(self.dataset_val,
                                batch_size=32,
                                shuffle=False,
                                num_workers=10,
                                pin_memory=True,
                                drop_last=True))

        return val_loaders

    def test_dataloader(self):
        #test_loaders = {
        #    name: DataLoader(self.dataset_test[name], batch_size=256, shuffle=False, num_workers=2)
        #    for name in self.dataset_test.keys()}
        test_loaders = [DataLoader(self.dataset_test[name],
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=5,
                                   pin_memory=True,)
                        for name in self.dataset_test.keys()]
        return test_loaders


class CifarCorrupted(Dataset):
    def __init__(self, base_c_path, corruption, level5_corruption_only=False,
                 transform=None):
        self.images = np.load(os.path.join(base_c_path, corruption + '.npy'))
        self.labels = torch.LongTensor(np.load(os.path.join(base_c_path, 'labels.npy')))
        if level5_corruption_only:
            self.images = self.images[-5000:]
            self.labels = self.labels[-5000:]

        self.transform = transform

    def __getitem__(self, index):
        # imgs = torch.from_numpy(self.images[index]).float()
        imgs = self.images[index]
        labels = self.labels[index]
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels

    def __len__(self):
        return len(self.labels)
