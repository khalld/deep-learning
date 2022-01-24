import torch
from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from os.path import splitext, join
from PIL import Image
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

def printer_helper(str):
    return print("***** %s *****" % str)

def split_into_train_and_test(dataset, train_size_perc=0.8):
    train_size = int(train_size_perc * len(dataset))
    test_size = len(dataset) - train_size

    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])

    return dataset_train, dataset_test

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

class TrashbinDataset(data.Dataset): # data.Dataset https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    """ A map-style dataset class used to manipulate a dataset composed by:
        image path of trashbin and associated label that describe the available capacity of the trashbin
            0 : empty trashbin
            1 : half trashbin
            2 : full trashbin

        Attributes
        ----------
        data : str
            path of csv file
        transform : torchvision.transforms

        Methods
        -------
        __len__()
            Return the length of the dataset

        __getitem__(i)
            Return image, label of i element of dataset  
    """

    def __init__(self, csv: str=None, transform: transforms=None, path_gdrive: str=''):
        """ Constructor of the dataset
            Parameters
            ----------
            csv : str
            path of the dataset

            transform : torchvision.transforms
            apply transform to the dataset

            path_gdrive: str
            necessary to apply the prepath in gdrive witouth changing csv

            Raises
            ------
            NotImplementedError
                If no path is passed is not provided a default dataset, default to load the image use only the csv file
        """
        
        if csv is None:
            raise NotImplementedError("No default dataset is provided")
        if splitext(csv)[1] != '.csv':
            raise NotImplementedError("Only .csv files are supported")
        
        self.data = pd.read_csv(csv)        # import from csv using pandas
        self.data = self.data.iloc[np.random.permutation(len(self.data))]       # random auto-permutation of the data
        self.transform = transform
        self.path_gdrive = path_gdrive

    def __len__(self):
        """ Return length of dataset """
        return len(self.data)

    def __getitem__(self, i=None):
        """ Return the i-th item of dataset

            Parameters
            ----------
            i : int
            i-th item of dataset

            Raises
            ------
            NotImplementedError
            If i is not a int
        """
        if i is None:
            raise NotImplementedError("Only int type is supported for get the item. None is not allowed")
        
        im_path, im_label = self.data.iloc[i]['image'], self.data.iloc[i].label
        im = Image.open(join(self.path_gdrive,im_path))        # Handle image with Image module from Pillow https://pillow.readthedocs.io/en/stable/reference/Image.html
        if self.transform is not None:
            im = self.transform(im)
        return im, im_label

class TripletTrashbin(data.Dataset):
    def __init__(self, root = 'dataset/all_labels.csv', transform = None) -> None:
#        super().__init__()
        self.dataset = TrashbinDataset(root, transform=transform)
        # self.dataset = self.dataset.data    # dipende dalla classe sopra, evito di chiamare un oggetto lungo
        self.class_to_indices = [np.where(self.dataset.data.label == label)[0] for label in range(3)]  # N delle classi

        self.generate_triplets()
    
    def generate_triplets(self):
        """ Genera le triplete associando ongi elemento del dataset due nuovi elementi. Uno simile e uno dissimile"""

        # verifico che class_to_indices funzioni
        # print(self.class_to_indices[0])
        # print(len(self.class_to_indices[0]))
        # print(len(self.class_to_indices[1]))
        # print(len(self.class_to_indices[2]))
        # print(len(self.class_to_indices[0]) + len(self.class_to_indices[1]) + len(self.class_to_indices[2]))
        # print(len(self.dataset))

        self.similar_idx = []
        self.dissimilar_idx = []

        printer_helper("Start making triplets...")
        for i in range(len(self.dataset)):
            # classe del primo elemento della tripletta
            c1 = self.dataset[i][1] # la classe la trovo sempre alla posizione 1 dato il dataset di sopra
            # indice dell'elemento simile
            j = np.random.choice(self.class_to_indices[c1])
            # scelgo una classe diversa a caso
            diff_class = np.random.choice(list(set(range(3))-{c1}))
            # campiono dalla classe di ^ per ottenere l'indice dell'elemento dissimile
            k = np.random.choice(self.class_to_indices[diff_class])

            self.similar_idx.append(j)
            self.dissimilar_idx.append(k)

        print("Triplets process ended")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        im1, l1 = self.dataset[index]
        im2, l2 = self.dataset[self.similar_idx[index]]
        im3, l3 = self.dataset[self.dissimilar_idx[index]]

        return im1, im2, im3, l1, l2, l3

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

class TripletNetworkTask(pl.LightningModule):
    def __init__(self, embedding_net, lr=0.01, momentum=0.99, margin=2, num_class=3):
        super(TripletNetworkTask, self).__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_net
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.num_class = num_class

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)

    def training_step(self, batch, batch_idx):
        I_i, I_j, I_k, *_ = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        # calcoliamo la loss
        l = self.criterion(phi_i, phi_j, phi_k)

        self.log('train/loss', l)
        return l

    def validation_step(self, batch, batch_idx):
        I_i, I_j, I_k, *_ = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        #calcolo la loss
        l = self.criterion(phi_i, phi_j, phi_k)
        self.log('valid/loss', l)

        if batch_idx == 0:
            self.logger.experiment.add_embedding(phi_i, batch[self.num_class], I_i, global_step = self.global_step)

#TODO: devi farlo diventare un autoencoder Conv
class EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1,32,5), 
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,stride=2),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(32,64,5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,stride=2))
        
        self.fc = nn.Sequential(nn.BatchNorm1d(64*4*4),
                                nn.Linear(64*4*4, 256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256,128))
    
    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        # Un semplice encoder con due layer fully connected
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Definiamo inoltre due layer lineari, uno per la media e uno per la varianza
        self.mean  = nn.Linear(hidden_dim, latent_dim)
        self.log_var   = nn.Linear(hidden_dim, latent_dim)
        #assumeremo che l'encoder dia in output il logaritmo della varianza
        
    def forward(self, x):
        #processiamo l'input attraverso i due layer fully connected
        h = self.backbone(x)
        #calcoliamo media e varianza
        mu = self.mean(h)
        log_var = self.log_var(h)
        
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        #anche in questo caso abbiamo due layer fully connected
        #seguiti da un layer di output
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        #processiamo l'input con l'MLP
        h = self.backbone(x)
        #applichiamo una attivazione di tipo
        #sigmoide per convertire gli output 
        #tra zero e uno (valori di grigio o probabilità)
        x_pred = torch.sigmoid(h)
        return x_pred

class VAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, beta):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, output_dim)
        self.beta = beta
        
    def reparameterization_trick(self, mean, std):
        #perform the reparametrization trick
        epsilon = std.new(std.shape).normal_() #generate a random epsilon value with the same size as mean and var
        #apply the trick to sample z
        z = mean + std*epsilon 
        return z
        
    def forward(self, x):
        #calcoliamo la media e il logaritmo della varianza
        mean, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)#calcoliamo la deviazione standard dal logarimo della varianza
        #campioniamo z mediante il reparametrization trick
        z = self.reparameterization_trick(mean, std)
        x_pred = self.decoder(z) #decodifichiamo il valore generato z
        
        #restituiamo la predizione, la media e il logaritmo della varianza
        return x_pred, mean, log_var
    
    # questo metodo definisce l'optimizer
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch #scartiamo le etichette
        x_pred, mean, log_var = self.forward(x)
        
        reconstruction_loss = F.binary_cross_entropy(x_pred, x, reduction='sum')
        kl_loss = - 0.5 * torch.sum(1+ log_var - mean**2 - log_var.exp())
        
        loss = reconstruction_loss + self.beta*kl_loss
        self.log('train/reconstruction_loss', reconstruction_loss.item())
        self.log('train/kl_loss', kl_loss.item())
        self.log('train/loss', loss.item())
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch #scartiamo le etichette
        x_pred, mean, log_var = self.forward(x)
        
        reconstruction_loss = F.binary_cross_entropy(x_pred, x, reduction='sum')
        kl_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        
        #calcoliamo la loss di validation come fatto per la loss di training
        loss = reconstruction_loss + self.beta*kl_loss
        self.log('val/reconstruction_loss', reconstruction_loss.item())
        self.log('val/kl_loss', kl_loss.item())
        self.log('val/loss', loss.item())
        
        #se questo è il primo batch, salviamo le immagini di input e quelle generate per dopo
        if batch_idx==0:
            return {'inputs': x, 'outputs': x_pred}
        
    def validation_epoch_end(self, results):
        images_in = results[0]['inputs'].view(-1,1,28,28)[:50,...]
        images_out = results[0]['outputs'].view(-1,1,28,28)[:50,...]
        self.logger.experiment.add_image('input_images', make_grid(images_in, nrow=10, normalize=True),self.global_step)
        self.logger.experiment.add_image('generated_images', make_grid(images_out, nrow=10, normalize=True),self.global_step)

if __name__ == "__main__":

    np.random.seed(1996)
    torch.manual_seed(1996)
    random.seed(1996)


    PATH_DST = join('dataset', 'all_labels.csv')
    PATH_GDRIVE = ''
    NUM_WORKERS = 8
    BATCH_SIZE = 1024
    NUM_EPOCHS = 1
    GPUS = 0
    
    # mean and dev std of MNIST
    mean = 0.1307
    std = 0.3081

    dataset_df = pd.read_csv(PATH_DST)

    dic_dst = {
        0: 'empty',
        1: 'half',
        2: 'full'
    }

    # TEST PER ADATTARE IL VAE AD IMMAGINI 256 X 256 a 3 canali (partendo da un modello 28 x 28 )
    transform = transforms.Compose([
                                    transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    torch.flatten
                                    ])

    dataset = TrashbinDataset(csv=PATH_DST, transform=transform)

    dataset_train, dataset_test = split_into_train_and_test(dataset)

    dataset_train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    dataset_test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    logger = TensorBoardLogger("vae_logs", name="fc_vae")

    trashbin_fc_vae = VAE(784, 512, 128, 784, beta=10)

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, gpus=GPUS, logger=logger) 
    trainer.fit(trashbin_fc_vae, dataset_train_loader, dataset_test_loader)