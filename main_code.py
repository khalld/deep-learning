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


if __name__ == "__main__":

    random.seed(1996)
    np.random.seed(1996)
    pl.seed_everything(1996)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    PATH_DST = 'dataset/all_labels.csv'
    PATH_GDRIVE = ''
    # TODO: se setto > 0 mi da 
    # [W ParallelNative.cpp:214] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
    # e non mi permette di effettuare il training. tuttavia resta troppo lento. come procedo?
    NUM_WORKERS = 0
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    GPUS = 0
    PRETRAINED_MODEL_PATH =  'models/squeezeNet_pretrained.pth'
    num_class = 3

    # valori pretrained
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    squeezeNet_1_0 = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    # applico le opportune modifiche
    squeezeNet_1_0.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1,1), stride=(1,1))
    # # # carico i pesi salvati

    squeezeNet_1_0.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))

    # testo così
    squeezeNet_1_0.classifier = nn.Sequential(
        # nn.Dropout(p=0.5, inplace=False),
        # nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1)),
        # nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Identity()
    )

    squeezeNet_1_0(torch.zeros(1, 3, 224,224)).shape

    transf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    dst_triplet = TripletTrashbin(root=PATH_DST, transform=transf)

    dst_train_triplet, dst_test_triplet = split_into_train_and_test(dst_triplet)

    triplet_dataset_train_loader = DataLoader(dst_train_triplet, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    triplet_dataset_test_loader = DataLoader(dst_test_triplet, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    triplet_trashbin_task = TripletNetworkTask(squeezeNet_1_0, lr=0.002)
    logger = TensorBoardLogger("metric_logs", name="test_trashbin_v1_1",)

    trainer = pl.Trainer(gpus=GPUS, logger = logger, max_epochs = 10, check_val_every_n_epoch = 5, )
    trainer.fit(triplet_trashbin_task, triplet_dataset_train_loader, triplet_dataset_test_loader)

    # Secondo training

    # triplet_trashbin_task_v2 = TripletNetworkTask(squeezeNet_1_0, lr=0.002)

    # checkpoint_callback = [ ModelCheckpoint(
    #     monitor= 'valid/loss',
    #     dirpath='/Users/danilo/GitHub/deep-learning/metric_logs/test_trashbin_v1/version_0/',
    #     filename='epoch=9-step=3299'
    # ) ]

    # logger = TensorBoardLogger("metric_logs", name="test_trashbin_v1_1",)
    # trainer = pl.Trainer(gpus=GPUS, logger = logger, max_epochs = 15, check_val_every_n_epoch = 1, callbacks=checkpoint_callback )
    # trainer.fit(triplet_trashbin_task_v2, triplet_dataset_train_loader, triplet_dataset_test_loader, ckpt_path='metric_logs/test_trashbin_v1/version_0/checkpoints/epoch=9-step=3299.ckpt')