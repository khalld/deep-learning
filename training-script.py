from libs.Dataset import TripletTrashbinDataModule
from libs.Model import TripletNetwork, TripletNetworkV2, evaluating_performance_and_save_tsne_plot, evaluating_performance_only
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
    MAX_EPOCHS = 31
    LOGS_FOLDER = "logs"
    MAIN_MODELS_FOLDER = "models"
    CKPT_LAST_PATH = "TripletMarginLoss-epoch-30.ckpt"
    
    dm = TripletTrashbinDataModule(img_size=DATA_IMG_SIZE,num_workers=N_WORKERS)
    dm.setup()

    dm_v2 = TripletTrashbinDataModule(img_size=DATA_IMG_SIZE,num_workers=N_WORKERS, trb_train_csv="triplet_training_v2.csv", trb_val_csv="triplet_validation_v2.csv", trb_test_csv="triplet_test_v2.csv")
    dm_v2.setup()

    dm_v3 = TripletTrashbinDataModule(img_size=DATA_IMG_SIZE,num_workers=N_WORKERS, trb_train_csv="triplet_training_v3.csv", trb_val_csv="triplet_validation_v3.csv", trb_test_csv="triplet_test_v3.csv")
    dm_v3.setup()


    # ---- Training Triplet Network with Triplet Margin Loss --------

    # tripletNetwork_tml = TripletNetwork()

    # Ho già allenato la rete con il dataset di default per 30 epoche, quindi lo carico
    tripletNetwork_tml = TripletNetwork.load_from_checkpoint(checkpoint_path=join(MAIN_MODELS_FOLDER, CKPT_LAST_PATH)) 

    logger_tml = TensorBoardLogger(join(MAIN_MODELS_FOLDER, LOGS_FOLDER), name="TripletMarginLoss")

    MAX_EPOCHS = MAX_EPOCHS + 15

    trainer1 = pl.Trainer(gpus=GPUS,
                        max_epochs=MAX_EPOCHS,
                        callbacks=[progress.TQDMProgressBar()],
                        logger=logger_tml,
                        accelerator="auto",
                        )

    trainer1.fit(model=tripletNetwork_tml, datamodule=dm_v2, ckpt_path=join(MAIN_MODELS_FOLDER, CKPT_LAST_PATH))
    trainer1.save_checkpoint(join(MAIN_MODELS_FOLDER, 'TripletMarginLoss-epoch{}.ckpt'.format(MAX_EPOCHS - 1)))
    torch.save(trainer1.model.state_dict(), join(MAIN_MODELS_FOLDER,'TripletMarginLoss-epoch-{}.pth'.format(MAX_EPOCHS - 1)))
    evaluating_performance_and_save_tsne_plot(tripletNetwork_tml, datamodule=dm, plot_name='TripletMarginLoss-epoch-{}-TSNE'.format(MAX_EPOCHS - 1))

    MAX_EPOCHS = MAX_EPOCHS + 15
    trainer1 = pl.Trainer(gpus=GPUS,
                    max_epochs=MAX_EPOCHS,
                    callbacks=[progress.TQDMProgressBar()],
                    logger=logger_tml,
                    accelerator="auto",
                    )
    
    trainer1.fit(model=tripletNetwork_tml, datamodule=dm_v3, ckpt_path=join(MAIN_MODELS_FOLDER, CKPT_LAST_PATH))
    trainer1.save_checkpoint(join(MAIN_MODELS_FOLDER, 'TripletMarginLoss-epoch{}.ckpt'.format(MAX_EPOCHS - 1)))
    torch.save(trainer1.model.state_dict(), join(MAIN_MODELS_FOLDER,'TripletMarginLoss-epoch-{}.pth'.format(MAX_EPOCHS - 1)))
    evaluating_performance_and_save_tsne_plot(tripletNetwork_tml, datamodule=dm, plot_name='TripletMarginLoss-epoch-{}-TSNE'.format(MAX_EPOCHS - 1))

    # ---- Training Triplet Network with Triplet Margin with Distance Loss --------

    # tripletNetwork_tmwd = TripletNetworkV2()
    # evaluating_performance_and_save_tsne_plot(tripletNetwork_tmwd, datamodule=dm, plot_name='TripletMarginWithDistanceLoss-epoch-{}-TSNE'.format(0))

    # tripletNetwork_tmwdl = TripletNetworkV2.load_from_checkpoint(checkpoint_path=join(MAIN_MODELS_FOLDER, CKPT_LAST_PATH)) 

    # logger_tml = TensorBoardLogger(join(MAIN_MODELS_FOLDER, LOGS_FOLDER), name="TripletMarginWithDistanceLoss")

    # trainer1 = pl.Trainer(gpus=GPUS,
    #                     max_epochs=MAX_EPOCHS,
    #                     callbacks=[progress.TQDMProgressBar()],
    #                     logger=logger_tml,
    #                     accelerator="auto",
    #                     )

    # trainer1.fit(model=tripletNetwork_tmwdl, datamodule=dm, ckpt_path=join(MAIN_MODELS_FOLDER, CKPT_LAST_PATH))
    # trainer1.save_checkpoint(join(MAIN_MODELS_FOLDER, 'TripletMarginWithDistanceLoss-epoch-{}.ckpt'.format(MAX_EPOCHS - 1)))
    # torch.save(trainer1.model.state_dict(), join(MAIN_MODELS_FOLDER,'TripletMarginWithDistanceLoss-epoch-{}.pth'.format(MAX_EPOCHS - 1)))
    
    # evaluating_performance_and_save_tsne_plot(tripletNetwork_tmwdl, datamodule=dm, plot_name='TripletMarginWithDistanceLoss-epoch-{}-TSNE'.format(MAX_EPOCHS - 1))