from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from os.path import splitext, join
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import transforms
import code as cu # custom utils

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

class TripletTrashbin(data.Dataset):
    def __init__(self, root = 'dataset/all_labels.csv', transform = None, path_gdrive='') -> None:
#        super().__init__()
        self.dataset = TrashbinDataset(root, transform=transform, path_gdrive=path_gdrive)
        # self.dataset = self.dataset.data    # dipende dalla classe sopra, evito di chiamare un oggetto lungo
        self.class_to_indices = [np.where(self.dataset.data.label == label)[0] for label in range(3)]  # N delle classi

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
            j = np.random.choice(self.class_to_indices[c1])
            # scelgo una classe diversa a caso
            diff_class = np.random.choice(list(set(range(3))-{c1}))
            # campiono dalla classe di ^ per ottenere l'indice dell'elemento dissimile
            k = np.random.choice(self.class_to_indices[diff_class])

            self.similar_idx.append(j)
            self.dissimilar_idx.append(k)

        # cu.printer_helper("Dataset loaded successfully!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        im1, l1 = self.dataset[index]
        im2, l2 = self.dataset[self.similar_idx[index]]
        im3, l3 = self.dataset[self.dissimilar_idx[index]]

        return im1, im2, im3, l1, l2, l3
