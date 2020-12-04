# imports
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import cv2

# pandas data organization
dataset_path = Path('@/datasets')
mask_path = dataset_path / 'AFDB_masked_face_dataset'
no_mask_path = dataset_path / 'AFDB_face_dataset'

mask_data_frame = pd.DataFrame()

# parse no mask images and set 0 to the "mask" value
for subj in tqdm(list(no_mask_path.iterdir()), desc="photos that have no masks on"):
    for image_path in subj.iterdir():
        img = cv2.imread(str(image_path))
        mask_data_frame = mask_data_frame.append({"image": img, "mask": 0}, ignore_index=True)

# parse mask images and set 1 to the "mask" value
for subj in tqdm(list(mask_path.iterdir()), desc="photos that have masks on"):
    for image_path in subj.iterdir():
        img = cv2.imread(str(image_path))
        mask_data_frame = mask_data_frame.append({"image": img, "mask": 1}, ignore_index=True)

mask_data_frame.to_pickle("@/data/mask_dataframe.pickle")


# build the dataset
# convert into Tensor so that PyTorch can manipulate it
class MaskSet(pd.Dataset()):

    def __init__(self, df):
        self.data_frame = df
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor()
        ])

    def __getitem__(self, key):
        row = self.data_frame.iloc[key]

        return {"image": self.transformations(row["image"]), "mask": tensor([row["mask"]], dtype=long)}

    def __len__(self):
        return len(self.data_frame.index)
