from torch.optim import SGD
from torch import nn
import pytorch_lightning as pl
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3,32,5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.BatchNorm2d(32),
                                        nn.Conv2d(32,64,5),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(
            nn.BatchNorm1d(64 * 4 * 4),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256,128))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

class TripletNetworkTask(pl.LightningModule):
    # lr uguale a quello del progetto vecchio
    def __init__(self, embedding_net, lr=0.002, momentum=0.99, margin=2, num_class=3):
        super(TripletNetworkTask, self).__init__()

        # TODO: credo debba essere escluso https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
        # self.save_hyperparameters()

        self.embedding_net = embedding_net
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.num_class = num_class
        self.lr = lr
        self.momentum = momentum

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # TODO: abilita uno o l'altro in base a se abiliti save_hyparameters()
        return SGD(self.embedding_net.parameters(), self.lr, momentum=self.momentum)
        # return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)

    # Lightning automatically sets the model to training for training_step and to eval for validation.
    def training_step(self, batch, batch_idx):

        I_i, I_j, I_k, *_ = batch

        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        # calcoliamo la loss
        # TODO: migliora
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        loss_embedd = phi_i.norm(2) + phi_i.norm(2) + phi_i.norm(2)

        print(f"loss embedd {loss_embedd}")

        # FIXME: dopo un po ritorna NaN con squeezenet1_1
        loss = loss_triplet + 0.001 *loss_embedd
        
        print(f"loss {loss}")

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        I_i, I_j, I_k, *_ = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        #calcolo la loss
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        
        loss_embedd = phi_i.norm(2) + phi_i.norm(2) + phi_i.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        self.log('valid/loss', loss)
        return loss