import numpy as np
from tqdm import tqdm
import faiss
import torch

def split_into_train_and_test(dataset, train_size_perc=0.8):
    train_size = int(train_size_perc * len(dataset))
    test_size = len(dataset) - train_size

    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])

    return dataset_train, dataset_test

def extract_rgb_representations(loader):
    """ Baseline basata su nearest neighbor RGB"""
    representations, labels = [], []
    for batch in tqdm(loader, total=len(loader)):
        representations.append(batch[0].view(batch[0].shape[0],-1).numpy())
        labels.append(batch[1])

    return np.concatenate(representations), np.concatenate(labels)

def predict_nn(train_rep, test_rep, train_label):
    """ Funzione che permette di predire le etichette sul test set analizzando l'algoritmo NN. !pip install faiss-gpu/cpu"""
    # inizializzo l'oggetto index utilizzato x indicizzare le rappresentazioni
    index = faiss.IndexFlat(train_rep.shape[1])
    # aggiungo le rappresentazioni di training all'indice
    index.add(train_rep.astype(np.float32))
    # effettuiamo la ricerca

    indices = np.array([index.search(x.reshape(1,-1).astype(np.float32), k=1)[1][0][0] for x in test_rep])

    #restituisco le etichette predette
    return train_label[indices].squeeze()

def evaluate_classification(pred_label, ground_truth):
    """ Valuto la bontà delle predizioni ottenute calcolando la distanza euclidea tra il vettore di label
        predetto e quelli di ground truth"""
    dist = np.sqrt(np.sum(np.square(pred_label-ground_truth)))
    return dist

def set_parameter_requires_grad(model, feature_extracting: bool):
    """Helper function that sets the `require_grad` attribute of parameter in the model to False when is used feature extracting"""

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def extract_rep_squeezeNet(model, loader, device="cpu"):
    """ Il modello estrae vettori di rappresentazione di 1280 unità. definisco una funzione per estrarre le rappresentazioni di dataloader di training e test """
    
    # Whenever you want to test your model you want to set it to model.eval() before which will disable dropout
    # (and do the appropriate scaling of the weights), also it will make batchnorm work on the averages computed
    # during training. Your code where you’ve commented model.eval() looks like like the right spot to set it to
    # evaluation mode. Then after you simply do model.train() and you’ve enabled dropout, batchnorm to work as previously.
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