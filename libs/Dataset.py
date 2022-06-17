import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils import data # necessary to create a map-style dataset https://pytorch.org/docs/stable/data.html
from os.path import splitext, join
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader
from torch.nn import ModuleList

class TripletTrashbinDataset(data.Dataset): # data.Dataset https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset
    def __init__(self, csv: str=None, transform: transforms=None):

        if csv is None:
            raise NotImplementedError("No default dataset is provided")
        if splitext(csv)[1] != '.csv':
            raise NotImplementedError("Only .csv files are supported")
        
        self.data = pd.read_csv(csv)
        self.data = remove_unnamed_col(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i=None):

        if i is None:
            raise NotImplementedError("Only int type is supported for get the item. None is not allowed")
        
        im_path_anchor, im_label_anchor = self.data.iloc[i]['anchor_image'], self.data.iloc[i].anchor_label
        im_anchor = Image.open(im_path_anchor)        # Handle image with Image module from Pillow https://pillow.readthedocs.io/en/stable/reference/Image.html
        if self.transform is not None:
            im_anchor = self.transform(im_anchor)

        im_path_pos, im_label_pos = self.data.iloc[i]['pos_image'], self.data.iloc[i].pos_label
        im_pos = Image.open(im_path_pos)
        if self.transform is not None:
            im_pos = self.transform(im_pos)

        im_path_neg, im_label_neg = self.data.iloc[i]['neg_image'], self.data.iloc[i].neg_label
        im_neg = Image.open(im_path_neg)
        if self.transform is not None:
            im_neg = self.transform(im_neg)

        return im_anchor, im_label_anchor, im_pos, im_label_pos, im_neg, im_label_neg

class TripletTrashbinDataModule(pl.LightningDataModule):
    def __init__(self, img_size, batch_size=32, num_workers=0, data_augmentation=True):
        super().__init__()

        self.batch_size = batch_size
        self.num_classes = 3
        self.img_size = img_size
        self.num_workers = num_workers

        self.trb_train_csv = 'dataset/triplet_training.csv'
        self.trb_val_csv = 'dataset/triplet_validation.csv'
        self.trb_test_csv = 'dataset/triplet_test.csv'
        self.data_augmentation = data_augmentation

        if data_augmentation:
            self.train_transform = transforms.Compose([
                transforms.Resize(self.img_size + 6),
                transforms.RandomCrop(self.img_size),
                # transforms.RandomApply(ModuleList([
                #     transforms.ColorJitter(brightness=.3, hue=.2),
                # ]), p=0.3),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
                # transforms.RandomEqualize(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            self.test_transform = transforms.Compose([
                transforms.Resize(self.img_size + 32), 
                transforms.CenterCrop(self.img_size),
                # transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
                # transforms.RandomInvert(p=0.3),
                # transforms.RandomHorizontalFlip(p=0.2),
                # transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        else:    
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])


    # def prepare_data(self):
        # do nothing

    def setup(self, stage: Optional[str] = None):

        if self.data_augmentation:
            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                self.trb_train = TripletTrashbinDataset(self.trb_train_csv, transform=self.train_transform)
                self.trb_val = TripletTrashbinDataset(self.trb_val_csv, transform=self.train_transform)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.trb_test = TripletTrashbinDataset(self.trb_test_csv, transform=self.test_transform)
        else:
            if stage == "fit" or stage is None:
                self.trb_train = TripletTrashbinDataset(self.trb_train_csv, transform=self.transform)
                self.trb_val = TripletTrashbinDataset(self.trb_val_csv, transform=self.transform)

            if stage == "test" or stage is None:
                self.trb_test = TripletTrashbinDataset(self.trb_test_csv, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trb_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.trb_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.trb_test, batch_size=self.batch_size, num_workers=self.num_workers)

def create_triplet_csv(all_labels_path=join("dataset", "all_labels.csv"), dest_csv_path=join("dataset", "all_labels_triplet.csv")):
    """
        Function that allows to arrange a triplet dataset to perform the task from the original.
        The original .csv with the dataset is available here: https://drive.google.com/drive/folders/1LmN-fXWZ8UpRkLeMjbootN46V9AHaE4x?usp=sharing (ask for permission)
    """
    triplet_df = pd.DataFrame({"anchor_image": [],
                        "anchor_label": [],
                        "pos_image": [],
                        "pos_label": [],
                        "neg_image": [],
                        "neg_label": []
                    })

    lb_csv = pd.read_csv(all_labels_path)
    class_dict = ['empty', 'half', 'full']

    lb_empty_csv = lb_csv.query('label == 0')
    lb_half_csv = lb_csv.query('label == 1')
    lb_full_csv = lb_csv.query('label == 2')

    print("Dataset dimension: empty: %d half: %d full: %d" % (len(lb_empty_csv), len(lb_half_csv), len(lb_full_csv) ))

    triplet_df = pd.DataFrame({"anchor_image": [],
                        "anchor_label": [],
                        "pos_image": [],
                        "pos_label": [],
                        "neg_image": [],
                        "neg_label": []
                    })

    for idx, row in tqdm(lb_csv.iterrows(), total=lb_csv.shape[0]):
        if (row['label'] == 0):

            # using .sample() an element of the dataframe is fished randomly
            pos_row = lb_empty_csv.sample()

            if np.random.choice((True, False)):
                neg_row = lb_half_csv.sample()
            else:
                neg_row = lb_full_csv.sample()

        elif (row['label'] == 1):
            pos_row = lb_half_csv.sample()

            if np.random.choice((True, False)):
                neg_row = lb_empty_csv.sample()
            else:
                neg_row = lb_full_csv.sample()
        else:
            pos_row = lb_full_csv.sample()

            if np.random.choice((True, False)):
                neg_row = lb_half_csv.sample()
            else:
                neg_row = lb_empty_csv.sample()

        triplet_df = triplet_df.append({"anchor_image": row['image'],
                        "anchor_label": row['label'],
                        "pos_image": pos_row.iloc[0,0],
                        "pos_label": pos_row.iloc[0,1],
                        "neg_image": neg_row.iloc[0, 0],
                        "neg_label": neg_row.iloc[0,1]
                    }, ignore_index=True)
        
    triplet_df = triplet_df.sample(frac=1).reset_index(drop=True)
    triplet_df.to_csv(dest_csv_path)

def split_train_val_test(dataset, perc):
    """
        Split dataset into training and test set using sklearn.model_selection.train_test_split function
    """
    train, testval = train_test_split(dataset, test_size = perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2]))
    return train, val, test

def remove_unnamed_col(df):
    """
        Utility function that returns the input DataFrame witouth the column 'Unnamed: 0'
    """
    res = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
    return res