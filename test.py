import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from libs.code import *
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
import faiss
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from typing import Optional
import pytorch_lightning as pl 
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

class FashionMNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 28 * 28 * 3
        self.conv1 = nn.Conv2d(1,16, stride = 1, padding = 1, kernel_size = 3)
        # 14 * 14 * 16
        self.conv2 = nn.Conv2d(16,32, stride = 1, padding = 1, kernel_size = 3)
        # 7 * 7 * 32
        self.conv3 = nn.Conv2d(32,64, stride = 1, padding = 1, kernel_size = 3)
        # 3 * 3 * 64

        self.fc1 = nn.Linear(3*3*64,128)
        self.fc2 = nn.Linear(128,64)
        self.out = nn.Linear(64,10)

        self.pool = nn.MaxPool2d(2,2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))

        batch_size, _, _, _ = x.size()

        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters())

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)

        # Logging the loss
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)

        # Logging the loss
        self.log('valid/loss', loss, on_epoch=True)
        return loss


class DataModuleFashionMNIST(pl.LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()

        self.download_dir = 'dataset_fashionmnist'
        self.dir = 'FashionMNIST'
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        datasets.FashionMNIST(self.dir, train = True, download = True)
        datasets.FashionMNIST(self.dir, train = False, download = True)

    def setup(self, stage=None):
        data = datasets.FashionMNIST(self.dir,
                                     train = True, 
                                     transform = self.transform)

        self.train, self.valid = random_split(data, [52000, 8000])

        self.test = datasets.FashionMNIST(self.download_dir,
                                          train = False,
                                          transform = self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size = self.batch_size, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers=12)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=12)

# https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
class LitClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

if __name__ == "__main__":
    # using datamodule
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    
    # mnist = MNISTDataModule("sample_dst")

    # trainer = pl.Trainer(max_epochs=1)
    # model = LitModel()

    # trainer.fit(model, datamodule=mnist)

    model = FashionMNISTModel()
    data = DataModuleFashionMNIST()
    logger = TensorBoardLogger("test_logs", name="test_trashbin_v1_1",)

    trainer = pl.Trainer(max_epochs=1 , logger = logger)

    trainer.fit(model, data)
