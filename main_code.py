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
    
    NUM_WORKERS = 12
    BATCH_SIZE = 256    # o 128
    NUM_EPOCHS = 10
    GPUS = 0
    PRETRAINED_MODEL_PATH =  'models/squeezeNet_pretrained.pth'
    num_class = 3

    # valori pretrained
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # TODO: Transformazioni che ho usato per il vecchio progetto.
    # Controllare se posso usarle per il ML
    # perché non mi funzionava il modello?

    # transf_train = transforms.Compose([
    #         transforms.Resize(230), # taglio solo una piccola parte col randomCrop in modo tale da prendere sempre il secchio
    #         transforms.RandomCrop(224),
    #         transforms.RandomApply(ModuleList([
    #             transforms.ColorJitter(brightness=.3, hue=.2),
    #         ]), p=0.3),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomHorizontalFlip(p=0.3),
    #         transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
    #         transforms.RandomEqualize(p=0.2),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=mean, std=std)
    #     ])

    # transf_test = transforms.Compose([
    #     transforms.Resize(256), 
    #     transforms.CenterCrop(224), 
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
    #     transforms.RandomInvert(p=0.3),
    #     transforms.RandomHorizontalFlip(p=0.2),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])


    # ----- carico dataset singolo

    # df = pd.read_csv(PATH_DST)

    # df_train, df_test = train_test_split(df, test_size=0.20, random_state=0)

    # print("df_train: {} , df_test: {}, is splitted correctly: {}".format(len(df_train), len(df_test), (len(df) == (len(df_test)+len(df_train)) )))

    # df_train.to_csv("dataset/df_training.csv")
    # df_test.to_csv("dataset/df_test.csv")

    # dst_train = TrashbinDataset(csv=PATH_DST, transform=transf)
    # dst_test = TrashbinDataset(csv=PATH_DST, transform=transf)

    # dst_train_loader = DataLoader(dst_train, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    # dst_test_loader = DataLoader(dst_test, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False)

    # ------- estrazione delle rappresentazioni e prime predizioni con nn

    # dst_train_rep_rgb, dst_train_labels = extract_rgb_representations(loader=dst_train_loader)
    # dst_test_rep_rgb, dst_test_labels = extract_rgb_representations(loader=dst_test_loader)

    # rappresentazioni di rtaining

    # dst_train_rep_rgb.shape

    #  ottengo le perdizioni sul test set usando predict_nn

    # pred_test_label_rgb = predict_nn(dst_train_rep_rgb, dst_test_rep_rgb, dst_train_labels)
    # print(f"Sample di label: {pred_test_label_rgb}")

    # valuto le performance delle baseline

    # classification_error = evaluate_classification(pred_test_label_rgb, dst_test_labels)
    # print(f"Classification error: {classification_error:0.2f}")

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

    # carico il dataset in triplette 
    dst_triplet = TripletTrashbin(root=PATH_DST, transform=transf)

    dst_train_triplet, dst_test_triplet = split_into_train_and_test(dst_triplet)

    triplet_dataset_train_loader = DataLoader(dst_train_triplet, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    triplet_dataset_test_loader = DataLoader(dst_test_triplet, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    # --- training

    triplet_trashbin_task = TripletNetworkTaskDebugged(squeezeNet_1_0, lr=0.002)
    logger = TensorBoardLogger("metric_logs", name="test_trashbin_v1_1",)

    trainer = pl.Trainer(gpus=GPUS, logger = logger, max_epochs = 10, check_val_every_n_epoch = 2 )
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