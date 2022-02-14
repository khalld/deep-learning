import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from libs.code import *
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
import faiss

from libs.Dataset import *
from libs.util import *
from libs.SiameseNetwork import TripletNetworkTask
# non necessari !
# from libs.code import *
# from libs.VAE import *
from sklearn.model_selection import train_test_split

PATH_DST = 'dataset/all_labels.csv'
PATH_GDRIVE = ''
NUM_WORKERS = 12
BATCH_SIZE = 256    # o 128
NUM_EPOCHS = 10
GPUS = 0
PRETRAINED_MODEL_PATH =  'models/squeezeNet_pretrained.pth'

if __name__ == "__main__":
    seed = random.seed(1996)
    np.random.seed(1996)
    pl.seed_everything(1996)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # valori pretrained
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # ----- carico dataset singolo

    df = pd.read_csv(PATH_DST)

    df_train, df_test = train_test_split(df, test_size=0.20, random_state=seed)

    print("df_train: {} , df_test: {}, is splitted correctly: {}".format(len(df_train), len(df_test), (len(df) == (len(df_test)+len(df_train)) )))

    df_train.to_csv("dataset/df_training.csv")
    df_test.to_csv("dataset/df_test.csv")

    dst_train = TrashbinDataset(csv=PATH_DST, transform=transf)
    dst_test = TrashbinDataset(csv=PATH_DST, transform=transf)

    dst_train_loader = DataLoader(dst_train, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    dst_test_loader = DataLoader(dst_test, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

    # ------- estrazione delle rappresentazioni e prime predizioni con nn
    print("Extracting rgb representations")
    dst_train_rep_rgb, dst_train_labels = extract_rgb_representations(loader=dst_train_loader)
    dst_test_rep_rgb, dst_test_labels = extract_rgb_representations(loader=dst_test_loader)

    print("Rappresentazioni di training {}".format(dst_train_rep_rgb.shape))

    #  ottengo le perdizioni sul test set usando predict_nn
    pred_test_label_rgb = predict_nn(dst_train_rep_rgb, dst_test_rep_rgb, dst_train_labels)
    print(f"Predicted test label rgb: {pred_test_label_rgb}")

    # valuto le performance delle baseline
    classification_error = evaluate_classification(pred_test_label_rgb, dst_test_labels)
    print(f"Classification error: {classification_error:0.2f}")
