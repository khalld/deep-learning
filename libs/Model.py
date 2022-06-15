import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.optim import SGD
from torch import nn
import torch
import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)
from tqdm import tqdm
import faiss
from sklearn.manifold import TSNE
from torchvision.models import squeezenet1_1
from os.path import splitext, join

class TripletNetwork(pl.LightningModule):
    """
        Modello triplet che allena SqueezeNet a ....
        
        Gli argomenti sono fissi per evitare errori durante il load del checkpoint.
    """
    def __init__(self, lr=7.585775750291837e-08, momentum=0.99, num_class=3, batch_size=256, criterion=nn.TripletMarginLoss(margin=2)):
        super(TripletNetwork, self).__init__()

        # TODO: Devi ancora provare se rimuovendo l'ignore ottieni risultati migliori
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['embedding_net'])

        squeezeNet = squeezenet1_1(pretrained=True)
        squeezeNet.classifier = nn.Identity()

        self.embedding_net = squeezeNet
        self.criterion = criterion

        self.num_class = num_class
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

    def forward(self, x):
        return self.embedding_net(x)

    def configure_optimizers(self):
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)

    # Lightning automatically sets the model to training for training_step and to eval for validation.
    def training_step(self, batch, batch_idx):
        I_i, _, I_j, _, I_k, _ = batch

        anchor = self.embedding_net(I_i)
        positive = self.embedding_net(I_j)
        negative = self.embedding_net(I_k)

        # calcolo la loss
        l = self.criterion(anchor, positive, negative)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('train/loss', l) #, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return l

    def validation_step(self, batch, batch_idx):
        I_i, _, I_j, _, I_k, _ = batch
        anchor = self.embedding_net(I_i)
        positive = self.embedding_net(I_j)
        negative = self.embedding_net(I_k)
        
        l = self.criterion(anchor, positive, negative)
        
        self.log('valid/loss', l)
        
        if batch_idx == 0:
            self.logger.experiment.add_embedding(anchor, batch[3], I_i, global_step=self.global_step)

def extr_rgb_rep(loader):
    """
        function that allows to extract rgb representations from a dataloader
    """
    representations, label = [], []
    for batch in tqdm(loader, total=len(loader)):
        representations.append(batch[0].view(batch[0].shape[0], -1).numpy())
        label.append(batch[1])

    return np.concatenate(representations), np.concatenate(label)

def extract_representation(model, loader):
    """
        TODO:
    """
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

def plot_values_tsne(embedding_net, test_loader):
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
    plt.show()

def evaluating_performance(lighting_module, datamodule):
    """
        Calcola l'errore di classificazione del modello, crea il TSNE e lo visualizza (per notebook jupiter)
    """
    # Uso il modello per estrarre le rappresentazione dal training e dal test_set

    train_rep_base, train_label = extract_representation(lighting_module, datamodule.train_dataloader())
    test_rep_base, test_label = extract_representation(lighting_module, datamodule.test_dataloader())

    # Valuto le performance del sistema con queste rappresentazioni non ancora ottimizzate

    pred_test_label_base = predict_nn(train_rep=train_rep_base, test_rep=test_rep_base, train_label=train_label)

    class_error = evaluate_classification(pred_test_label_base, test_label)

    print('Classification error {}'.format(class_error))

    plot_values_tsne(lighting_module.embedding_net, datamodule.test_dataloader())


def evaluating_performance_and_save_tsne_plot(lighting_module, datamodule, plot_name=""):
    """
        Calcola l'errore di classificazione del modello, crea il TSNE e lo salva su file
    """
    # Uso il modello per estrarre le rappresentazione dal training e dal test_set

    train_rep_base, train_label = extract_representation(lighting_module, datamodule.train_dataloader())
    test_rep_base, test_label = extract_representation(lighting_module, datamodule.test_dataloader())

    # Valuto le performance del sistema con queste rappresentazioni non ancora ottimizzate

    pred_test_label_base = predict_nn(train_rep=train_rep_base, test_rep=test_rep_base, train_label=train_label)

    class_error = evaluate_classification(pred_test_label_base, test_label)
    print('Classification error {}'.format(class_error))

    test_rep, test_labels = extract_representation(lighting_module.embedding_net, datamodule.test_dataloader())
    selected_rep = np.random.choice(len(test_rep), 10000)
    selected_test_rep = test_rep[selected_rep]
    selected_test_labels = test_labels[selected_rep]
    
    tsne = TSNE(2)
    rep_tsne = tsne.fit_transform(selected_test_rep)

    plt.figure(figsize=(8,6))
    for c in np.unique(selected_test_labels):
        plt.plot(rep_tsne[selected_test_labels==c, 0], rep_tsne[selected_test_labels==c, 1], 'o', label=c)
    plt.legend()
    plt.savefig(join('models/', plot_name))