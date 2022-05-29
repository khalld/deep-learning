import torch
from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from os.path import splitext, join
import numpy as np
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch import nn
from torch.optim import SGD
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from pytorch_lightning.loggers import TensorBoardLogger
import random
from random import choice
from torch.optim import Adam
from torch.nn import functional as F
import torch.optim as optim

# DEPRECATI! SE LI DEVI UTILIZZARE METTILI IN UTIL!!

def printer_helper(str):
    return print("***** %s *****" % str)

def reverse_norm(image):
    """Allow to show a normalized image"""
    
    image = image-image.min()
    return image/image.max()

def extract_codes(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    codes, labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        code, *_ = model(x)
        # code = code.detach().to('cpu').numpy() # TODO only if is possible use GPU
        code = code.detach().numpy()
        labels.append(batch[1])
        codes.append(code)
    return np.concatenate(codes), np.concatenate(labels)

def extract_representations(model, loader, device):
    model.eval()
    model.to(device)
    representations, labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        rep = model(x)
        rep = rep.detach().to(device).numpy()
        labels.append(batch[1])
        representations.append(rep)
    return np.concatenate(representations, np.concatenate(labels))

def make_TSNE(autoencoder: pl.LightningModule, test_loader: DataLoader) -> None:
    codes, labels = extract_codes(autoencoder, test_loader)
    print(codes.shape, labels.shape)

    # trasformo le mappe di feature in vettori monodimensionali e seleziono un sottoinsieme di dati
    selected_codes = np.random.choice(len(codes),1000)
    codes = codes.reshape(codes.shape[0],-1)
    codes = codes[selected_codes]
    labels = labels[selected_codes]
    print(codes.shape)

    # trasformo i dati mediante TSNE ed eseguo il plot
    tsne = TSNE(2)
    codes_tsne_conv=tsne.fit_transform(codes)
    plt.figure(figsize=(8,6))
    for c in np.unique(labels):
        plt.plot(codes_tsne_conv[labels==c, 0], codes_tsne_conv[labels==c, 1], 'o', label= c)
    plt.legend()
    plt.show()
