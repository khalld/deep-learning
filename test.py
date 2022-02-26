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


        # self.data_dir = data_dir
        # self.data_csv = pd.read_csv(os.path.join(self.data_dir, csv))
                # import from csv using pandas
        self.transform = transforms.Compose([    transforms.Grayscale(num_output_channels=1), transforms.Resize((28,28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

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
