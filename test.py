from posixpath import dirname
from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from os.path import splitext, join
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
import code as cu # custom utils
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import deprecation
from torch.nn import ModuleList
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models import squeezenet1_1
from torch import nn
import torch
from torch.optim import SGD
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, progress
from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.models import squeezenet1_1
from torch import nn
import torch
from torchvision.models import mobilenet_v2
# or to ignore all warnings that could be false positives
import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)
from tqdm import tqdm
import faiss
from sklearn.manifold import TSNE

class TripletTrashbinDataset(data.Dataset): # data.Dataset https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    def __init__(self, csv: str=None, transform: transforms=None):

        if csv is None:
            raise NotImplementedError("No default dataset is provided")
        if splitext(csv)[1] != '.csv':
            raise NotImplementedError("Only .csv files are supported")
        
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i=None):

        if i is None:
            raise NotImplementedError("Only int type is supported for get the item. None is not allowed")
        
        im_path_anchor, im_label_anchor = self.data.iloc[i]['anchor_image'], self.data.iloc[i].anchor_label
        im_anchor = Image.open(im_path_anchor)        # Handle image with Image module from Pillow https://pillow.readthedocs.io/en/stable/reference/Image.html
        if self.transform is not None:
            im_anchor = self.transform(im_anchor)

        im_path_pos, im_label_pos = self.data.iloc[i]['pos_image'], self.data.iloc[i].pos_label
        im_pos = Image.open(im_path_pos)
        if self.transform is not None:
            im_pos = self.transform(im_pos)

        im_path_neg, im_label_neg = self.data.iloc[i]['neg_image'], self.data.iloc[i].neg_label
        im_neg = Image.open(im_path_neg)
        if self.transform is not None:
            im_neg = self.transform(im_neg)

        return im_anchor, im_label_anchor, im_pos, im_label_pos, im_neg, im_label_neg

class TripletTrashbinDataModule(pl.LightningDataModule):
    def __init__(self, img_size, batch_size=32, num_workers=0):
        super().__init__()

        self.batch_size = batch_size
        self.num_classes = 3
        self.img_size = img_size
        self.num_workers = num_workers

        self.trb_train_csv = 'dataset/triplet_training.csv'
        self.trb_val_csv = 'dataset/triplet_validation.csv'
        self.trb_test_csv = 'dataset/triplet_test.csv'

        self.transform = transforms.Compose([
                        transforms.Resize((self.img_size, self.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        # transforms.Lambda(lambda x: x.view(-1))
                    ])


    # def prepare_data(self):
    #     # TODO: genera train e val randomicamente, al momento li tieni fissi così risparmi memoria
    #     print("Do nothing on prepare_data")

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:          
            self.trb_train = TripletTrashbinDataset(self.trb_train_csv, transform=self.transform)
            self.trb_val = TripletTrashbinDataset(self.trb_val_csv, transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.trb_test = TripletTrashbinDataset(self.trb_test_csv, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trb_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.trb_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.trb_test, batch_size=self.batch_size, num_workers=self.num_workers)

class TripletNetworkTask(pl.LightningModule):
    # lr uguale a quello del progetto vecchio
    def __init__(self, embedding_net, lr=0.002, momentum=0.99, margin=2, num_class=3, batch_size=32):
        super(TripletNetworkTask, self).__init__()

        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['embedding_net'])
        self.embedding_net = embedding_net
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.num_class = num_class
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

    def forward(self, x):
        return self.embedding_net(x)

    def configure_optimizers(self):
        # Dovrei mettere hparams.lr o self.lr?
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)
        # return SGD(self.embedding_net.parameters(), self.lr, momentum=self.hparams.momentum)

    # Lightning automatically sets the model to training for training_step and to eval for validation.
    def training_step(self, batch, batch_idx):
        I_i, _, I_j, _, I_k, _ = batch

        anchor = self.embedding_net(I_i)
        positive = self.embedding_net(I_j)
        negative = self.embedding_net(I_k)

        # calcolo la loss
        l = self.criterion(anchor, positive, negative)

        self.log('train/tripletMargin', l)
        
        return l

    def validation_step(self, batch, batch_idx):
        I_i, _, I_j, _, I_k, _ = batch
        anchor = self.embedding_net(I_i)
        positive = self.embedding_net(I_j)
        negative = self.embedding_net(I_k)
        
        l = self.criterion(anchor, positive, negative)
        
        self.log('valid/tripletMargin', l)
        
        if batch_idx == 0:
            self.logger.experiment.add_embedding(anchor, batch[3], I_i, global_step=self.global_step)

class TripletNetworkTaskV2(pl.LightningModule):
    # lr uguale a quello del progetto vecchio
    def __init__(self, embedding_net, lr=0.002, momentum=0.99, margin=2, num_class=3):
        super(TripletNetworkTask, self).__init__()

        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['embedding_net'])
        self.embedding_net = embedding_net
        # TODO: prova anche con L1Loss o MSELoss
        self.criterion = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function= nn.PairwiseDistance())
        self.num_class = num_class
        self.lr = lr
        self.momentum = momentum
        self.batch_size = 32

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Dovrei mettere hparams.lr o self.lr?
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)
        # return SGD(self.embedding_net.parameters(), self.lr, momentum=self.hparams.momentum)

    # Lightning automatically sets the model to training for training_step and to eval for validation.
    def training_step(self, batch, batch_idx):
        I_i, _, I_j, _, I_k, _ = batch

        anchor = self.embedding_net(I_i)
        positive = self.embedding_net(I_j)
        negative = self.embedding_net(I_k)

        # calcolo la loss
        l = self.criterion(anchor, positive, negative)

        self.log('train/tripletMargin', l)
        
        return l

    def validation_step(self, batch, batch_idx):
        I_i, _, I_j, _, I_k, _ = batch
        anchor = self.embedding_net(I_i)
        positive = self.embedding_net(I_j)
        negative = self.embedding_net(I_k)
        
        l = self.criterion(anchor, positive, negative)
        
        self.log('valid/tripletMargin', l)
        
        if batch_idx == 0:
            self.logger.experiment.add_embedding(anchor, batch[3], I_i, global_step=self.global_step)

def extr_rgb_rep(loader):
    representations, label = [], []
    for batch in tqdm(loader, total=len(loader)):
        representations.append(batch[0].view(batch[0].shape[0], -1).numpy())
        label.append(batch[1])

    return np.concatenate(representations), np.concatenate(label)

def extract_representation(model, loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    representations, labels = [], []
    for batch in tqdm(loader, total=len(loader)):
        x = batch[0].to(device)
        rep = model(x)
        rep = rep.detach().to('cpu').numpy()
        labels.append(batch[1])
        representations.append(rep)
    return np.concatenate(representations), np.concatenate(labels)

def predict_nn(train_rep, test_rep, train_label):
    """Funzione che permette di predire le etichette sul test set utilizzando NN"""
    index = faiss.IndexFlatL2(train_rep.shape[1])

    index.add(train_rep.astype(np.float32))

    indices = np.array([index.search(x.reshape(1,-1).astype(np.float32), k=1)[1][0][0] for x in test_rep])

    return train_label[indices].squeeze()

def evaluate_classification(pred_label, gt_label):
    """Misuro la bontà delle predizioni ottenute calcolando la distanza euclidea tra i valori dei label predetti e quelli di groundt truth"""
    # classification_error = np.sqrt((pred_label - gt_label)**2).sum(1).mean()
    classification_error = np.sqrt(np.sum(np.square(pred_label-gt_label)))

    return classification_error

def plot_values_tsne(embedding_net, test_loader, figpath = "tsne"):
    test_rep, test_labels = extract_representation(embedding_net, test_loader)
    selected_rep = np.random.choice(len(test_rep), 10000)
    selected_test_rep = test_rep[selected_rep]
    selected_test_labels = test_labels[selected_rep]
    
    tsne = TSNE(2)
    rep_tsne = tsne.fit_transform(selected_test_rep)

    plt.figure(figsize=(8,6))
    for c in np.unique(selected_test_labels):
        plt.plot(rep_tsne[selected_test_labels==c, 0], rep_tsne[selected_test_labels==c, 1], 'o', label=c)
    plt.legend()
    # plt.show()
    plt.savefig('{}.png'.format(figpath))

# TODO: la devi testare per vedere se si comporta correttamente
def evaluating_performance(lighting_module, datamodule, bt_s):
    train_rep_base, train_label = extract_representation(lighting_module, datamodule.train_dataloader())
    test_rep_base, test_label = extract_representation(lighting_module, datamodule.test_dataloader())

    # Valuto le performance del sistema con queste rappresentazioni non ancora ottimizzate

    pred_test_label_base = predict_nn(train_rep=train_rep_base, test_rep=test_rep_base, train_label=train_label)

    class_error = evaluate_classification(pred_test_label_base, test_label)

    print('Classification error before training {}'.format(class_error))

    plot_values_tsne(lighting_module.embedding_net, datamodule.test_dataloader(), 'tsne_{}_batch'.format(bt_s))

if __name__ == "__main__":

    dic = {
        0: 'empty',
        1: 'half',
        2: 'full'
    }

    data_img_size = 224
    data_batch_size = 32

    #TODO: prova anche con 1, cerca eventualmente il migliore n di workers per il tuo pc su google
    n_workers = 0  # os.cpu_count()

    dm = TripletTrashbinDataModule(img_size=data_img_size,num_workers=n_workers)
    # dm.prepare_data()
    dm.setup()

    # TODO: verifica prestazioni prima del training

    mobileNet_v2 = mobilenet_v2()

    mobileNet_v2.classifier = nn.Identity()

    print("**** required for mobilenet_v2: {} ****".format( mobileNet_v2(torch.zeros(1,3,data_img_size,data_img_size)).shape))

    # TODO: Prova con batch_size 128 e 256 dato che hai provato che vanno bene entrambi!
    triplet_mobileNet = TripletNetworkTask(mobileNet_v2, lr=0.00000001, batch_size=data_batch_size)

    # **** Verifico usando la libreria la migliore dimensione per il batch *****

    # trainer = pl.Trainer(auto_scale_batch_size="power", max_epochs=-1)
    # trainer.tune(triplet_mobileNet, datamodule=dm)

    # **** verifico usando la libreria per trovare il mgliore LR ****

    # trainer = pl.Trainer(auto_lr_find=True)
    # lr_finder = trainer.tune(triplet_mobileNet, datamodule=dm)
    # TODO: trova il modo di fare il grafico del LR trovato!
    # triplet_mobileNet.hparams.lr = new_lr # TODO: da verificare se si deve fare così o col costruttore

    # **** predict-nn ****

    print("***** Evaluating performance before training *****")

    # Uso il modello non ancora allenato per estrarre le rappresentazione dal training e dal test_set

    train_rep_base, train_label = extract_representation(triplet_mobileNet, dm.train_dataloader())
    test_rep_base, test_label = extract_representation(triplet_mobileNet, dm.test_dataloader())

    # Valuto le performance del sistema con queste rappresentazioni non ancora ottimizzate

    pred_test_label_base = predict_nn(train_rep=train_rep_base, test_rep=test_rep_base, train_label=train_label)

    class_error = evaluate_classification(pred_test_label_base, test_label)

    print('Classification error before training {}'.format(class_error))

    plot_values_tsne(triplet_mobileNet.embedding_net, dm.test_dataloader(), 'tsne_{}_batch'.format(data_batch_size))

    # ***** Training del modello :

    logger = TensorBoardLogger("metric_logs", name="siamese_mobilenet_v1")

    trainer = pl.Trainer(gpus=0,
                        max_epochs=2,
                        callbacks=[progress.TQDMProgressBar()],
                        logger=logger,
                        accelerator="auto",
                        )

    trainer.fit(model=triplet_mobileNet, datamodule=dm)
    trainer.save_checkpoint('ckpt_backup/triplet_mobilenet_{}_batch'.format(data_batch_size) )

    # **** verifica prestazioni, TSNE e salva grafico dopo il training ****

    print("***** Evaluating performance after training *****")

    # TODO: fai un copia incolla brutale

    # **** Eventuale loading da checkpoint ****    

    # restoring training state --  If you don’t just want to load weights, but instead restore the full training, do the following:
    # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # trainer.fit(triplet_mobileNet, ckpt_path="some/path/to/my_checkpoint.ckpt")

    # TODO: Aggiungi loading dal checkpoint ed effettua il training per un totale di 10 epoche
    
    # TODO: fai altre epoche con una loss diversa!



