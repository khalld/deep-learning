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
    def __init__(self, embedding_net, lr=0.002, momentum=0.99, margin=2, num_class=3):
        super(TripletNetworkTask, self).__init__()

        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['embedding_net'])
        self.embedding_net = embedding_net
        self.criterion = nn.TripletMarginLoss(margin=margin)
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

# TODO: verifica sia corretta
def predict_nn(train_rep, test_rep, train_label):
    index = faiss.IndexFlatL2(train_rep.shape[1])

    index.add(train_rep.astype(np.float32))

    indices = np.array([index.search(x.reshape(1,-1).astype(np.float32), k=1)[1][0][0] for x in test_rep])

    return train_label[indices].squeeze()

def evaluate_classification(pred_label, gt_label):
    # TODO: definisci una funzione di valutazione, trovala
    return np.NaN

if __name__ == "__main__":

    dic = {
        0: 'empty',
        1: 'half',
        2: 'full'
    }

    data_img_size = 224

    #TODO: prova anche con 1, cerca eventualmente il migliore n di workers per il tuo pc su google
    n_workers = 0  # os.cpu_count()

    dm = TripletTrashbinDataModule(img_size=data_img_size,num_workers=n_workers)
    # dm.prepare_data()
    dm.setup()

    # TODO: verifica prestazioni prima del training

    mobileNet_v2 = mobilenet_v2()

    mobileNet_v2.classifier = nn.Identity()

    print("**** required for mobilenet_v2: {} ****".format( mobileNet_v2(torch.zeros(1,3,224,224)).shape))

    triplet_mobileNet = TripletNetworkTask(mobileNet_v2, lr=0.00000001)

    # **** Verifico usando la libreria la migliore dimensione per il batch *****

    # trainer = pl.Trainer(auto_scale_batch_size="power", max_epochs=-1)
    # trainer.tune(triplet_mobileNet, datamodule=dm)

    # **** verifico usando la libreria per trovare il mgliore LR ****

    # trainer = pl.Trainer(auto_lr_find=True)
    # lr_finder = trainer.tune(triplet_mobileNet, datamodule=dm)
    # TODO: trova il modo di fare il grafico del LR trovato!
    # triplet_mobileNet.hparams.lr = new_lr # TODO: da verificare se si deve fare così o col costruttore

    # TODO: **** predict-nn ****
    # TODO: **** verifica prestazioni dopo ****

    logger = TensorBoardLogger("metric_logs", name="siamese_mobilenet_v1")

    trainer = pl.Trainer(gpus=0,
                        max_epochs=2,
                        callbacks=[progress.TQDMProgressBar()],
                        logger=logger,
                        accelerator="auto",
                        )

    trainer.fit(model=triplet_mobileNet, datamodule=dm)



