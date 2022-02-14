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
import warnings
class TripletNetworkTaskDebugged(pl.LightningModule):
    # lr uguale a quello del progetto vecchio
    def __init__(self, embedding_net, lr=0.002, momentum=0.99, margin=2, num_class=3):
        super(TripletNetworkTaskDebugged, self).__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_net
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.num_class = num_class

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)

    # Lightning automatically sets the model to training for training_step and to eval for validation.
    def training_step(self, batch, batch_idx):

        # print("STEP 0: ")

        I_i, I_j, I_k, *_ = batch

        # print(f"i_i: {len(I_i)}, i_j :{len(I_j)}, i_k:{len(I_k)}")

        # print(f"Shape: {I_i.shape}")

        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        # print(f"phi_i: {phi_i}, phi_j :{phi_j}, phi_k:{phi_k}")

        # calcoliamo la loss
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        # print(f"training_step: loss_triplet {loss_triplet}")
        # self.log('train/loss', loss_triplet)
        # return loss_triplet
        
        loss_embedd = phi_i.norm(2) + phi_i.norm(2) + phi_i.norm(2)

        # print(f"loss embedd {loss_embedd}")

        loss = loss_triplet + 0.001 *loss_embedd
        
        # print(f"loss {loss}")

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        I_i, I_j, I_k, *_ = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        #calcolo la loss
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        # print(f"validation_step: loss_triplet {loss_triplet}")
        # self.log('train/loss', loss_triplet)
        # return loss_triplet
        loss_embedd = phi_i.norm(2) + phi_i.norm(2) + phi_i.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        self.log('valid/loss', loss)
        return loss

if __name__ == "__main__":
    # from https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    
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

    NUM_WORKERS = 0 # https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
    BATCH_SIZE = 128
    PRETRAINED_MODEL_PATH =  'models/squeezeNet_pretrained.pth'
    num_class = 3

    # valori pretrained
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # TODO: Transformazioni che ho usato per il vecchio progetto.
    # Controllare se posso usarle per il ML
    # perché non mi funzionava il modello?
    transf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # ---- carico il mio modello custom -------

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

    # --------------- carico il dataset in triplette 
    dst_triplet = TripletTrashbin(root=PATH_DST, transform=transf)

    dst_train_triplet, dst_test_triplet = split_into_train_and_test(dst_triplet)

    triplet_dataset_train_loader = DataLoader(dst_train_triplet, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    triplet_dataset_test_loader = DataLoader(dst_test_triplet, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    # --- training

    triplet_trashbin_task = TripletNetworkTaskDebugged(squeezeNet_1_0, lr=0.002)
    logger = TensorBoardLogger("metric_logs", name="test_trashbin_v1_1",)

    # TODO: Documentati se puoi fare di meglio !
    trainer = pl.Trainer(accelerator="cpu",
                        logger = logger,
                        max_epochs = 10,
                        check_val_every_n_epoch = 1,
                        log_every_n_steps=20,
                        )
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