import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from libs.code import *
from pytorch_lightning.loggers import TensorBoardLogger
# ------
# to export in ipynb
import torch
import torch.utils.data as data
from torchvision import transforms
import pytorch_lightning as pl
from libs.code import *
# custom libraries
from libs.Dataset import *
# -------


def get_train_images(num, dataset_train):
    return torch.stack([dataset_train[i][0] for i in range(num)], dim=0)

class Encoder(nn.Module):
    
    def __init__(self, 
                num_input_channels : int, 
                base_channel_size : int, 
                latent_dim : int, 
                act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        _TEMP = 256 # old value: 16
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16 || 128X128 => 64 x 64
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8 || 64 x 64 => 32x32
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4 || 32x32 => 16 x 16 == 256
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2* _TEMP *c_hid, latent_dim)
        )

    def forward(self, x):
        # print("PASSO A -- ENCODER")
        # print(x.shape)
        # print("**** ENCODER END \n\n")
        return self.net(x)

class Decoder(nn.Module):
    
    def __init__(self, 
                num_input_channels : int, 
                base_channel_size : int, 
                latent_dim : int, 
                act_fn : object = nn.GELU):
        """
        Inputs: 
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        _TEMP = 256 # old value 16
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * _TEMP * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

        # print("****************")
        # print(self.net)
    
    def forward(self, x):

        # print("PASSO A - INIT DECODER")
        # print(x.shape)
        x = self.linear(x)
        # print("PASSO B - LINEAR")
        # print(x.shape)

        _NEW_VALUE_TEMP = 16 # l'ultimo layer sopra era 4 x 4 noi lo voglaimo 16 x 16 dopo  le modifiche

        x = x.reshape(x.shape[0], -1, _NEW_VALUE_TEMP, _NEW_VALUE_TEMP)
        # print("PASSO C -- RESHAPE")
        # print(x.shape)
        x = self.net(x)
        # print("PASSO D")
        # print(x.shape)
        # print("**** DECODER END \n\n")

        return x

class Autoencoder(pl.LightningModule):
    
    def __init__(self, 
                base_channel_size: int, 
                latent_dim: int, 
                encoder_class : object = Encoder,
                decoder_class : object = Decoder,
                num_input_channels: int = 3, 
                width: int = 128, 
                height: int = 128):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters() 
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)
        
    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        # print("\n\n\nINIT FORWARD OF autoencoder *******\n\n\n")

        # print("X : ")
        # print(x.shape)
        # print("\n\n\n")
        z = self.encoder(x)
        # print("Z code : ")
        # print(z.shape)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', 
                                                        factor=0.2, 
                                                        patience=20, 
                                                        min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)                             
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)

class GenerateCallback(pl.Callback):
    
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)
        
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

def train_autoencoder(dataset_train, train_loader:DataLoader, test_loader:DataLoader, val_loader:DataLoader, latent_dim:int, checkpoint_path: str, save_dir: str, name: str, epochs:int =10, base_channel_size:int=32,):
    # Create a PyTorch Lightning trainer with the generation callback

    _logger = TensorBoardLogger(save_dir=save_dir, name=f"{name}_{latent_dim}")

    GPUS = 1 if str(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")).startswith("cuda") else 0

    print("******* Testing..... current GPU: %d **********" % GPUS)

    trainer = pl.Trainer(default_root_dir=os.path.join(save_dir, f"{name}_{latent_dim}"), 
                        gpus=GPUS, 
                        max_epochs=epochs, 
                        callbacks=[
                            # TODO: fatto per test
                                ModelCheckpoint(save_weights_only=True),    # default save checkpoint every 10 times
                                GenerateCallback(get_train_images(8, dataset_train=dataset_train), every_n_epochs=5),
                                LearningRateMonitor("epoch")], # TODO: aggiungine alte
                        logger=_logger)
                        

    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path)

    if os.path.isfile(pretrained_filename) and len(pretrained_filename) > 0:
        print("Found pretrained model, loading from: %s" % ( f"{name}_{latent_dim}.ckpt" ))
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=base_channel_size, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result

if __name__ == "__main__":

    # W = 16 # input size
    # K = 3 # filter size
    # P = 1 # stride
    # S = 2   # padding

    # (((W - K + 2*P)/S) + 1)
    #         # Here W = Input size
    #         # K = Filter size
    #         # S = Stride
    #         # P = Padding

    # Setting the seed
    pl.seed_everything(42)

    # do seeed.... np

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print("Device:", device)

    PATH_DST = 'dataset/all_labels.csv'
    PATH_GDRIVE = ''
    NUM_WORKERS = 8 # ricordati se significava tutto o niente
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    GPUS = 0

    # print(Autoencoder(base_channel_size=32, latent_dim=384).example_input_array.shape)

    transform = transforms.Compose([
                                transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                # torch.flatten # trasforma il tensore ad una dimensione
                                ])

    dataset = TrashbinDataset(csv=PATH_DST, transform=transform)

    # TODO: fixa la funzione!
    dataset_train, dataset_test = split_into_train_and_test(dataset)
    _, dataset_val = split_into_train_and_test(dataset)

    train_loader = data.DataLoader(dataset_train, batch_size=2, num_workers=NUM_WORKERS, shuffle=True)
    val_loader = data.DataLoader(dataset_val, batch_size=2, num_workers=NUM_WORKERS, shuffle=False)
    test_loader = data.DataLoader(dataset_test, batch_size=2, num_workers=NUM_WORKERS, shuffle=False)


    # print(train_loader.batch_sampler)
    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    # TODO: Testo separatamente encoder e decoder perché non mi funziona
    # funziona
    # model = Encoder(num_input_channels=3, base_channel_size=32, latent_dim=384)
    # print(model(torch.zeros(384,3,128,128)).shape,)
    # funziona
    # model = Decoder(num_input_channels=3, base_channel_size=32, latent_dim=384)
    # print(model(torch.zeros(384, 384)).shape,)
    # TODO: vedo cosa vuole l'autoencoder in input || TODO: DA CAPIRE PERCHPé VUOLE BATCH_SIZE 2
    # TODO: ***** NON FUNZIONA IL TRAINING ****************
    # temp = Autoencoder(base_channel_size=32, latent_dim=384)
    # print(temp.example_input_array.shape)

    model_dict = {}
    # for latent_dim in [64, 128, 256, 384]:
    for latent_dim in [384]:
        model_ld, result_ld = train_autoencoder(dataset_train = dataset_train, train_loader=train_loader, val_loader=val_loader,
                                                test_loader=test_loader, latent_dim=latent_dim, save_dir='logs/vae', checkpoint_path='',
                                                name="VAE_test_", epochs=NUM_EPOCHS, base_channel_size=32)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}