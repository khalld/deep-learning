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
from random import choice

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

if __name__ == "__main__":

    PATH_DST = join('dataset', 'all_labels.csv')
    PATH_GDRIVE = ''
    NUM_WORKERS = 8
    BATCH_SIZE = 1024
    NUM_EPOCHS = 5
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

    # ***** Show some part of dataframe *****
    # plt.figure(figsize=(15,8))
    # for ii, i in enumerate(np.random.choice(range(len(dataset_df)), 10)):
    #     plt.subplot(2,5,ii+1)
    #     plt.title("Class: %s" % dic_dst[dataset_df['label'][i]])
    #     plt.imshow(plt.imread(dataset_df['image'][i]),cmap='gray')
    # plt.show()


    # ***** Calculate mean and std *****
    # means = np.zeros(3)
    # stdevs = np.zeros(3)

    # for data in dataset_df:
    #     img = data[0]
    #     for i in range(3):
    #         img = np.asarray(img)
    #         means[i] += img[i, :, :].mean()
    #         stdevs[i] += img[i, :, :].std()

    # means = np.asarray(means) / dataset_df.__len__()
    # stdevs = np.asarray(stdevs) / dataset_df.__len__()
    # print("{} : normMean = {}".format(type, means))
    # print("{} : normstdevs = {}".format(type, stdevs))

    # transform = transforms.Compose([
    #                                 transforms.Grayscale(num_output_channels=1), #TODO: immagini in bianco e nero x semplificare e farlo uguale al prof
    #                                 transforms.Resize((28,28)),     # resize dell'immagine come in LAB 01 per fare i test TODO da adattare
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((mean,),(std)),
    #                                 ])

    # dataset = TrashbinDataset('dataset/all_labels.csv', transform=transform)

    # print("dataset len: %i" % len(dataset))
    # # print(dataset.data)   # verifico la permutazione su tutte le label già implementata con il resto della classe

    # # splitto il dataset in training e test senza considerare il validaiton
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # #validation_size =
    # dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])

    # print("train_size: %i" % (len(dataset_train)))
    # print("test_size: %i" % (len(dataset_test)))

    # # dataset_loader = DataLoader(dataset, batch_size=32)
    # dataset_train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    # dataset_test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # #dataset_validation_loader = ...

    # # controllo che la shape sia corettamente 28x28
    # # for a in dataset_train_loader:
    # #     print(a[0].shape)
    # #     break
    

    # printer_helper("START TO MAKE TRAINING")

    # logger = TensorBoardLogger("tb_logs", name="convolutional_autoencoder")
    # convolutional_autoencoder = AutoencoderConv()
    # trainer = pl.Trainer(max_epochs=NUM_EPOCHS, gpus=GPUS, logger=logger)
    # trainer.fit(convolutional_autoencoder, dataset_train_loader, dataset_test_loader)

    # printer_helper("END TRAINING")

    # make_TSNE(convolutional_autoencoder, dataset_test_loader)


    dataset_triplet = TripletTrashbin()
    
    # ***** Visualizzo la rete triplet implmentata ***** TODO: fai meglio
    
    # plt.figure(figsize=(18,4))
    # for ii, i in enumerate(np.random.choice(range(len(dataset_triplet)), 3)):
    #     plt.subplot(3, 10, ii+1)
    #     # plt.text(3,10, 'Main element %i' % (i))
    #     plt.imshow(dataset_triplet[i][0],)
        
    #     # plt.text(3,10, 'Similar to %i' % (i))
    #     plt.subplot(3, 10, ii+11)
    #     plt.imshow(dataset_triplet[i][1])

    #     # plt.text(3,10, 'Dissimilar to %i' % (i))
    #     plt.subplot(3, 10, ii+21)
    #     plt.imshow(dataset_triplet[i][2])
    # plt.show()