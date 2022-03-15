from libs.Dataset import *
from libs.SiameseNetwork import *
import matplotlib.pyplot as plt
import dill # allow to save notebook session, useful for test https://towardsdatascience.com/how-to-restore-your-jupyter-notebook-session-dfeadbd86d65
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, progress
from torch.optim import SGD
from torch import nn
import torch
from tqdm import tqdm


# TODO: FARE UN TEST CON LE FUNZIONI DEBUGGATE PER FARE IL PREDICT NN ED ESTRARRE LE FUNZIONI

def extract_rgb_rep(loader):
    """ Baseline che effettua ricerca nearest neighbor sui valori RGB grezzi.
    (ricerca l'immagine visivamente più simile)"""
    representations, labels = [], []
    for batch in tqdm(loader, total=len(loader)):
        # TODO: guarda screenshot
        print(batch.size())
        print(batch.shape())
        print(batch)
        representations.append(batch[0].view(batch[0].shape[0],-1).numpy())
        labels.append(batch[1])

    return np.concatenate(representations), np.concatenate(labels)


# DEF EXTRACT REP WITH MODEL!




if __name__ == "__main__":

    # load datamodule sessions
    dm = "datamodule"
    dill.load_session('notebook_env_28.db')

    print("Dims datamodel: {}".format(dm.dims))

    sample_dl = dm.val_dataloader()

    repres, labels = extract_rgb_rep(sample_dl)