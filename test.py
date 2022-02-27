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
from PIL import Image
from libs.Dataset import *
class CustomModel(pl.LightningModule): 
    def __init__(self): 
        super(CustomModel, self).__init__() 
        
        # Defining our model architecture
        self.fc1 = nn.Linear(28*28, 256) 
        self.fc2 = nn.Linear(256, 128) 
        self.out = nn.Linear(128, 10) 
        
        # Defining learning rate
        self.lr = 0.01
        
        # Defining loss 
        self.loss = nn.CrossEntropyLoss() 
    
    def forward(self, x):
        
        # Defining the forward pass of the model
        batch_size, _, _, _ = x.size() 
        x = x.view(batch_size, -1) 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        return self.out(x) 
    
    def configure_optimizers(self):
        
        # Defining and returning the optimizer for our model
        # with the defines parameters
        return torch.optim.SGD(self.parameters(), lr = self.lr) 
    
    def training_step(self, train_batch, batch_idx): 
        
        # Defining training steps for our model
        x, y = train_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y) 
        return loss 
    
    def validation_step(self, valid_batch, batch_idx): 
        
        # Defining validation steps for our model
        x, y = valid_batch 
        logits = self.forward(x) 
        loss = self.loss(logits, y)

class TrashbinDataModule(pl.LightningDataModule):
    def __init__(self, csv = 'all_labels.csv', data_dir: str = "/dataset/"):
        super().__init__()

        if csv is None:
            raise NotImplementedError("No default dataset is provided")
        if splitext(csv)[1] != '.csv':
            raise NotImplementedError("Only .csv files are supported")

        # import from csv using pandas
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def prepare_data(self):
        print("Do nothing...")

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.trb_train = TrashbinDataset('dataset/training.csv', transform=self.transform)
            self.trb_val = TrashbinDataset('dataset/validation.csv', transform=self.transform)

            # Optionally...
            self.dims = tuple(self.trb_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.trb_test = TrashbinDataset('dataset/test.csv', transform=self.transform)

            # Optionally...
            self.dims = tuple(self.trb_test[0][0].shape)

    # def setup(self, stage:)

    def train_dataloader(self):
        return DataLoader(self.trb_train, batch_size=32, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.trb_val, batch_size=32, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.trb_test, batch_size=32, num_workers=12)


class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

if __name__ == "__main__":
    
    dm = TrashbinDataModule()
    dm.setup()

    dm.prepare_data()
    dm.setup(stage="fit")

    model = CustomModel()
    # model = Model(num_classes=dm.num_classes, width=dm.width, vocab=dm.vocab)

    trainer = pl.Trainer(max_epochs=1)

    trainer.fit(model, dm)

    # TODO: AGGIUNGI PROJECTORS - LOGS - PRELOADING del dataset
