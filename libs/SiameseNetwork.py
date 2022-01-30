from torch.optim import SGD
from torch import nn
import pytorch_lightning as pl

class TripletNetworkTask(pl.LightningModule):
    # lr uguale a quello del progetto vecchio
    def __init__(self, embedding_net, lr=0.002, momentum=0.99, margin=2, num_class=3):
        super(TripletNetworkTask, self).__init__()
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
        I_i, I_j, I_k, *_ = batch

        print(f"i_i: {I_i}, i_j :{I_j}, i_k:{I_k}")

        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        phi_k = self.embedding_net(I_k)

        print(f"phi_i: {phi_i}, phi_j :{phi_j}, phi_k:{phi_k}")

        # calcoliamo la loss
        loss_triplet = self.criterion(phi_i, phi_j, phi_k)
        print(f"loss triplet {loss_triplet}")
        
        loss_embedd = phi_i.norm(2) + phi_i.norm(2) + phi_i.norm(2)

        print(f"loss embedd {loss_embedd}")

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
