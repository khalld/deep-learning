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

class TrashbinDataset(data.Dataset): # data.Dataset https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    """ A map-style dataset class used to manipulate a dataset composed by:
        image path of trashbin and associated label that describe the available capacity of the trashbin
            0 : empty trashbin
            1 : half trashbin
            2 : full trashbin

        Attributes
        ----------
        data : str
            path of csv file
        transform : torchvision.transforms

        Methods
        -------
        __len__()
            Return the length of the dataset

        __getitem__(i)
            Return image, label of i element of dataset  
    """

    def __init__(self, csv: str=None, transform: transforms=None, path_gdrive: str=''):
        """ Constructor of the dataset
            Parameters
            ----------
            csv : str
            path of the dataset

            transform : torchvision.transforms
            apply transform to the dataset

            path_gdrive: str
            necessary to apply the prepath in gdrive witouth changing csv

            Raises
            ------
            NotImplementedError
                If no path is passed is not provided a default dataset, default to load the image use only the csv file
        """
        
        if csv is None:
            raise NotImplementedError("No default dataset is provided")
        if splitext(csv)[1] != '.csv':
            raise NotImplementedError("Only .csv files are supported")
        
        self.data = pd.read_csv(csv)        # import from csv using pandas
        self.data = self.data.iloc[np.random.permutation(len(self.data))]       # random auto-permutation of the data
        self.transform = transform
        self.path_gdrive = path_gdrive

    def __len__(self):
        """ Return length of dataset """
        return len(self.data)

    def __getitem__(self, i=None):
        """ Return the i-th item of dataset

            Parameters
            ----------
            i : int
            i-th item of dataset

            Raises
            ------
            NotImplementedError
            If i is not a int
        """
        if i is None:
            raise NotImplementedError("Only int type is supported for get the item. None is not allowed")
        
        im_path, im_label = self.data.iloc[i]['image'], self.data.iloc[i].label
        im = Image.open(join(self.path_gdrive,im_path))        # Handle image with Image module from Pillow https://pillow.readthedocs.io/en/stable/reference/Image.html
        if self.transform is not None:
            im = self.transform(im)
        return im, im_label

class TrashbinDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=12):
        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_classes = 3
        self.num_workers = num_workers
        self.img_size = 28

        # import from csv using pandas
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def prepare_data(self):
        print("Do nothing...")

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            
            # TODO : per velocizzare il loading delle immagini. Puoi prevedere di caricare solo 'all_labels' e poi splittare di volta in volta 
            # randomicamente
            self.trb_train = TrashbinDataset(join(self.data_dir,'training.csv'), transform=self.transform)
            self.trb_val = TrashbinDataset(join(self.data_dir,'validation.csv'), transform=self.transform)

            # Optionally...
            self.dims = tuple(self.trb_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.trb_test = TrashbinDataset(join(self.data_dir,'test.csv'), transform=self.transform)

            # Optionally...
            self.dims = tuple(self.trb_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.trb_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.trb_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.trb_test, batch_size=self.batch_size, num_workers=self.num_workers)

class TripletTrashbin(data.Dataset):
    def __init__(self, root, transform = None, path_gdrive='') -> None:
        self.dataset = TrashbinDataset(root, transform=transform, path_gdrive=path_gdrive)
        self.class_to_idx = [np.where(self.dataset.data.label == label)[0] for label in range(3)]  # N delle classi

        self.generate_triplets()
    
    def generate_triplets(self):
        """ Genera le triplete associando ongi elemento del dataset due nuovi elementi. Uno simile e uno dissimile"""

        self.similar_idx = []
        self.dissimilar_idx = []

        # cu.printer_helper("Start making triplets...")

        for i in range(len(self.dataset)):
            # classe del primo elemento della tripletta
            c1 = self.dataset[i][1] # la classe la trovo sempre alla posizione 1 dato il dataset di sopra
            # indice dell'elemento simile
            j = np.random.choice(self.class_to_idx[c1])
            # scelgo una classe diversa a caso
            diff_class = np.random.choice(list(set(range(3))-{c1}))
            # campiono dalla classe di ^ per ottenere l'indice dell'elemento dissimile
            k = np.random.choice(self.class_to_idx[diff_class])

            self.similar_idx.append(j)
            self.dissimilar_idx.append(k)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        im1, l1 = self.dataset[index]
        im2, l2 = self.dataset[self.similar_idx[index]]
        im3, l3 = self.dataset[self.dissimilar_idx[index]]

        return im1, im2, im3, l1, l2, l3

class TripletTrashbinDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=12):
        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_classes = 3
        self.img_size = 224
        self.num_workers = num_workers

        self.trb_all_csv = 'all_labels.csv'
        self.trb_train_csv = 'training.csv'
        self.trb_val_csv = 'validation.csv'
        self.trb_test_csv = 'test.csv'

        # import from csv using pandas
        # FIXME: DEPRECATA
        # self.transform = transforms.Compose([
        #     transforms.Resize((self.img_size,self.img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        #     )
        # ])

        self.train_transform=transforms.Compose([
                transforms.Resize(230), # taglio solo una piccola parte col randomCrop in modo tale da prendere sempre il secchio
                transforms.RandomCrop(224),
                transforms.RandomApply(ModuleList([
                    transforms.ColorJitter(brightness=.3, hue=.2),
                ]), p=0.3),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
                transforms.RandomEqualize(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.test_transform=transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
                transforms.RandomInvert(p=0.3),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


    def prepare_data(self):
        # TODO: genera train e val randomicamente, al momento li tieni fissi così risparmi memoria
        print("Do nothing on prepare_data")

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.trb_train = TripletTrashbin(join(self.data_dir,self.trb_train_csv), transform=self.train_transform)
            self.trb_val = TripletTrashbin(join(self.data_dir,self.trb_val_csv), transform=self.train_transform)
            
            # Optionally...
            self.dims = tuple(self.trb_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.trb_test = TripletTrashbin(join(self.data_dir,self.trb_test_csv), transform=self.test_transform)

            # Optionally...
            self.dims = tuple(self.trb_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.trb_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.trb_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.trb_test, batch_size=self.batch_size, num_workers=self.num_workers)

