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

def extract_representations(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    representations, labels = [], []
    for batch in loader:
        x = batch[0].to(device)
        rep = model(x)
        rep = rep.detach().to('cpu').numpy()
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

class AutoencoderConv(pl.LightningModule):
    """Autoencoder basato su convoluzioni
        È tutto uguale ad un autoencoder FC eccetto il
        costruttore e validation_epoch_end
    
    """
    def __init__(self):
        super(AutoencoderConv, self).__init__()

        self.encoder = nn.Sequential(nn.Conv2d(1,16,3, padding=1),
                                        nn.AvgPool2d(2),
                                        nn.ReLU(),
                                        nn.Conv2d(16,8,3, padding=1),
                                        nn.AvgPool2d(2),
                                        nn.ReLU(),
                                        nn.Conv2d(8,4,3, padding=1),
                                        nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(4,8,3, padding=1),
                                        nn.Upsample(scale_factor=2),
                                        nn.ReLU(),
                                        nn.Conv2d(8,16,3, padding=1),
                                        nn.Upsample(scale_factor=2),
                                        nn.ReLU(),
                                        nn.Conv2d(16,1,3, padding=1))
        # loss utilizzata per il training
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        code = self.encoder(x)
        reconstructed = self.decoder(code)
        # restituisco sia il codice che l'output ricostruito
        return code, reconstructed

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.99)
        return optimizer
    
    # questo metodo definisce come effettuare ogni singolo step di training
    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        _, reconstructed = self.forward(x)
        loss = self.criterion(x, reconstructed)
        self.log('train/loss', loss)
        return loss
    
    # questo metodo definisce come effettuare ogni singolo step di validation
    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        _, reconstructed = self.forward(x)
        loss = self.criterion(x, reconstructed)
        self.log('val/loss', loss)
        if batch_idx==0:
            return {'inputs':x, 'outputs':reconstructed}

    def validation_epoch_end(self, results):
        images_in = results[0]['inputs'].view(-1,1,28,28)[:50,...]
        images_out = results[0]['outputs'].view(-1,1,28,28)[:50,...]
        self.logger.experiment.add_image('input_images', make_grid(images_in, nrow=10, normalize=True),self.global_step)
        self.logger.experiment.add_image('generated_images', make_grid(images_out, nrow=10, normalize=True),self.global_step)

# if __name__ == "__main__":

#     np.random.seed(1996)
#     torch.manual_seed(1996)
#     random.seed(1996)


#     PATH_DST = '../dataset/all_labels.csv'
#     PATH_GDRIVE = ''
#     NUM_WORKERS = 8
#     BATCH_SIZE = 1024
#     NUM_EPOCHS = 1
#     GPUS = 0
    
#     # mean and dev std of MNIST
#     mean = 0.1307
#     std = 0.3081

#     dataset_df = pd.read_csv(PATH_DST)

#     dic_dst = {
#         0: 'empty',
#         1: 'half',
#         2: 'full'
#     }

#     # TEST PER ADATTARE IL VAE AD IMMAGINI 256 X 256 a 3 canali (partendo da un modello 28 x 28 )
#     transform = transforms.Compose([
#                                     # transforms.Grayscale(num_output_channels=1),
#                                     transforms.Resize((28,28)),
#                                     transforms.ToTensor(),
#                                     torch.flatten
#                                     # torch.flatten # trasforma il tensore ad una dimensione
#                                     ])

#     dataset = TrashbinDataset(csv=PATH_DST, transform=transform, path_gdrive='..')

#     dataset_train, dataset_test = split_into_train_and_test(dataset)

#     dataset_train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
#     dataset_test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

#     # mi controllo la shape del dataloader per calcolarmi i layer in formato rgb
#     # for i, data in enumerate(dataset_train_loader, 0):
#     #     # get the inputs
#     #     inputs, labels = data
#     #     inputs = np.array(inputs)
#     #     print(inputs.shape)
#     #     # Run your training process
#     #     print(f'Epoch: {i} | Inputs {inputs.shape} | Labels {labels}')
#     #     break

#     # input shape con grayscale n output = 1 --> (1024, 784)
#     # input shape               normale      --> (1024, 2352)


#     logger = TensorBoardLogger("vae_logs", name="fc_vae")

#     # grayscale | torch. flatten TODO: vedi se entrambi i dataloader con grayscale n output=1 e torch.flatten danno lo stesso risultato
#     # trashbin_fc_vae = VAE(784, 512, 128, 784, beta=10)

#     trashbin_fc_vae = VAE(784 * 3, 512 * 3, 128 * 3, 784 * 3, beta=10)
#     trainer = pl.Trainer(max_epochs=NUM_EPOCHS, gpus=GPUS, logger=logger) 
#     trainer.fit(trashbin_fc_vae, dataset_train_loader, dataset_test_loader)