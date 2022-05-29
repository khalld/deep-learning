import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm, tqdm_pandas

import pandas as pd
import numpy as np

if __name__ == "__main__":

    triplet_df = pd.DataFrame({"anchor_image": [],
                        "anchor_label": [],
                        "pos_image": [],
                        "pos_label": [],
                        "neg_image": [],
                        "neg_label": []
                    })

    lb_csv = pd.read_csv('dataset/all_labels.csv')

    class_dict = ['empty', 'half', 'full']

    lb_empty_csv = lb_csv.query('label == 0')
    lb_half_csv = lb_csv.query('label == 1')
    lb_full_csv = lb_csv.query('label == 2')

    print("Dataset dimension:")
    print("empty: %d half: %d full: %d" % (len(lb_empty_csv), len(lb_half_csv), len(lb_full_csv) ))

    triplet_df = pd.DataFrame({"anchor_image": [],
                        "anchor_label": [],
                        "pos_image": [],
                        "pos_label": [],
                        "neg_image": [],
                        "neg_label": []
                    })

    for idx, row in tqdm(lb_csv.iterrows(), total=lb_csv.shape[0]):

        # print('(image: {}, label: {},'.format(row['image'], row['label']))
        # print('(image: {}, label: {},'.format(row['image'], class_dict[row['label']]))

        # TODO: volevi realizzare un modo per prendere il record random ed rimuoverlo sempre,
        # ma in questo modo non stai riuscendo ad eliminare l'idx selezionato quindi pisci

        # empty_idx = np.random.choice(lb_empty_csv.index, 1, replace=False)[0]
        # half_idx = np.random.choice(lb_half_csv.index, 1, replace=False)[0]
        # full_idx = np.random.choice(lb_full_csv.index, 1, replace=False)[0]

        if (row['label'] == 0):
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
    triplet_df.to_csv('dataset/all_labels_triplet.csv')