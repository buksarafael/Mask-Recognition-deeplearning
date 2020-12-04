from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

dataset_path = Path('datasets')
mask_path = dataset_path / 'AFDB_masked_face_dataset'
no_mask_path = dataset_path / 'AFDB_face_dataset'

mask_data_frame = pd.DataFrame()

# parse no mask images and set 0 to the "mask" value
for subj in tqdm(list(no_mask_path.iterdir()), desc="photos that have no masks on"):
    for image_path in subj.iterdir():
        mask_data_frame = mask_data_frame.append({"image": str(image_path), "mask": 0}, ignore_index=True)

# parse mask images and set 1 to the "mask" value
for subj in tqdm(list(mask_path.iterdir()), desc="photos that have masks on"):
    for image_path in subj.iterdir():
        mask_data_frame = mask_data_frame.append({"image": str(image_path), "mask": 1}, ignore_index=True)

data_frame_name = "data/mask_dataframe.pickle"
print(f"Save dataframe to: {data_frame_name}")
mask_data_frame.to_pickle("data/mask_dataframe.pickle")
