# imports
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
from torch import long, tensor
import numpy as np
import cv2


class MaskSet(Dataset):

    def __init__(self, data_frame):
        self.data_frame = data_frame

        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor()
        ])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')

        row = self.data_frame.iloc[key]
        image = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        return {"image": self.transformations(image), "mask": tensor([row["mask"]], dtype=long)}

    def __len__(self):
        return len(self.data_frame.index)
