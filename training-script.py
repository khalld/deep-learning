from libs.Dataset import TripletTrashbinDataModule
from libs.Model import TripletNetwork, evaluating_performance
from torchvision.models import squeezenet1_1
import torch
from torch import nn
from torch.optim import SGD
import pytorch_lightning as pl
from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from os.path import splitext, join
from torchvision import transforms
from typing import Optional
from torch.utils.data import DataLoader
from torch.nn import ModuleList
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import progress
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import faiss
import sklearn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import warnings # or to ignore all warnings that could be false positives

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    DATA_IMG_SIZE = 224
    DATA_BATCH_SIZE = 256
    N_WORKERS = 0  # os.cpu_count() , if != 0 return warning
    GPUS = 0
    LR = 7.585775750291837e-08
    MAX_EPOCHS = 30

    dm = TripletTrashbinDataModule(img_size=DATA_IMG_SIZE,num_workers=N_WORKERS)
    dm.setup()

    squeezeNet = squeezenet1_1(pretrained=True)
    squeezeNet.classifier = nn.Identity()

    # ---- Training Triplet Network with Triplet Margin Loss --------

    tripletNetwork_tml = TripletNetwork(squeezeNet, lr=LR, batch_size=DATA_BATCH_SIZE)
    logger_tml = TensorBoardLogger("models/logs", name="tripletNetwork_TripletMarginLoss")

    trainer1 = pl.Trainer(gpus=GPUS,
                        max_epochs=MAX_EPOCHS,
                        callbacks=[progress.TQDMProgressBar()],
                        logger=logger_tml,
                        accelerator="auto",
                        )

    trainer1.fit(model=tripletNetwork_tml, datamodule=dm)
    trainer1.save_checkpoint('models/tripletNetwork_TripletMarginLoss.ckpt')
    torch.save(trainer1.model.state_dict(), 'models/tripletNetwork_TripletMarginLoss.pth')

    # ---- Training Triplet Network with Triplet Margin with Dinstance Loss --------

    tripletNetwork_tmwdl = TripletNetwork(squeezeNet, lr=LR, batch_size=DATA_BATCH_SIZE, criterion=nn.TripletMarginWithDistanceLoss(margin=2, distance_function= nn.PairwiseDistance()))

    logger_tmwdl = TensorBoardLogger("models/logs", name="tripletNetwork_TripletMarginWithDistanceLoss")

    trainer2 = pl.Trainer(gpus=GPUS,
                        max_epochs=MAX_EPOCHS,
                        callbacks=[progress.TQDMProgressBar()],
                        logger=logger_tmwdl,
                        accelerator="auto",
                        )

    trainer2.fit(model=tripletNetwork_tmwdl, datamodule=dm)
    trainer2.save_checkpoint('models/tripletNetwork_TripletMarginWithDistanceLoss.ckpt')
    torch.save(trainer2.model.state_dict(), 'models/tripletNetwork_TripletMarginWithDistanceLoss.pth')