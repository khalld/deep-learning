from libs.Dataset import TripletTrashbinDataModule
from libs.Model import TripletNetwork, evaluating_performance_and_save_tsne_plot
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
    MAX_EPOCHS = 32
    LOGS_FOLDER = "logs"
    MAIN_MODELS_FOLDER = "models"

    # dm = TripletTrashbinDataModule(img_size=DATA_IMG_SIZE,num_workers=N_WORKERS)
    # dm.setup()

    # ---- Training Triplet Network with Triplet Margin Loss --------

    # NOTA: Stai verificando che save_hyperparameters senza ignore funzioni meglio di quello con ignore.
    # ... credo di sì!

    # tripletNetwork_tml = TripletNetwork() # il LR e l'embedding net è di default per evitare il problema del loading del checkpoint
    # tripletNetwork_tml = TripletNetwork.load_from_checkpoint('models/tripletNetwork_TripletMarginLoss.ckpt')
    # logger_tml = TensorBoardLogger(join(MAIN_MODELS_FOLDER, LOGS_FOLDER), name="tripletNetwork_TripletMarginLoss")

    # trainer1 = pl.Trainer(gpus=GPUS,
    #                     max_epochs=MAX_EPOCHS,
    #                     callbacks=[progress.TQDMProgressBar()],
    #                     logger=logger_tml,
    #                     accelerator="auto",
    #                     )

    # trainer1.fit(model=tripletNetwork_tml, datamodule=dm, ckpt_path='models/tripletNetwork_TripletMarginLoss.ckpt')
    # trainer1.save_checkpoint(join(MAIN_MODELS_FOLDER, 'tripletNetwork_TripletMarginLoss_v2.ckpt'))
    # torch.save(trainer1.model.state_dict(), join(MAIN_MODELS_FOLDER,'tripletNetwork_TripletMarginLoss_v2.pth'))
    
    # evaluating_performance_and_save_tsne_plot(tripletNetwork_tml, datamodule=dm, plot_name='test_{}-epochs'.format(MAX_EPOCHS))


    # ---- Training Triplet Network with Triplet Margin with Dinstance Loss --------

    # TODO: verifica se è necessario implementare un'altra classe per evitare i problemi del load_checkpoint

    # tripletNetwork_tmwdl = TripletNetwork(squeezeNet, lr=LR, batch_size=DATA_BATCH_SIZE, criterion=nn.TripletMarginWithDistanceLoss(margin=2, distance_function= nn.PairwiseDistance()))

    # logger_tmwdl = TensorBoardLogger("models/logs", name="tripletNetwork_TripletMarginWithDistanceLoss")

    # trainer2 = pl.Trainer(gpus=GPUS,
    #                     max_epochs=MAX_EPOCHS,
    #                     callbacks=[progress.TQDMProgressBar()],
    #                     logger=logger_tmwdl,
    #                     accelerator="auto",
    #                     )

    # trainer2.fit(model=tripletNetwork_tmwdl, datamodule=dm)
    # trainer2.save_checkpoint('models/tripletNetwork_TripletMarginWithDistanceLoss.ckpt')
    # torch.save(trainer2.model.state_dict(), 'models/tripletNetwork_TripletMarginWithDistanceLoss.pth')


    # -------- evaluating performance e test snza data augmentation per provare che funzioni correttamente senza

    # dm = TripletTrashbinDataModule(img_size=DATA_IMG_SIZE,num_workers=N_WORKERS, data_augmentation=False)
    # dm.setup()
    # tripletNetwork_tml = TripletNetwork.load_from_checkpoint('models/logs-no-dataaug/triplet_squeezeNet_v1/version_2/checkpoints/epoch=30-step=6417.ckpt')
    # evaluating_performance_and_save_tsne_plot(tripletNetwork_tml, datamodule=dm, plot_name='test_{}-epochs-NO-DATAAUG'.format(MAX_EPOCHS))
