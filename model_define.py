from typing import List, Union, Dict
import pytorch_lightning as pl
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, init, CrossEntropyLoss
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from maskdetection import MaskSet


# ------defining the model------
# using 4 convolutional layers
# using 2 linear layers
# ReLU as activation function
# xavier_uniform - make the network train better

class DetectorTrainer(pl.LightningModule):
    def __init__(self, mask_path: Path = None):
        super(DetectorTrainer, self).__init__()
        self.mask_path = mask_path
        self.mask_data_frame = None
        self.train_data_frame = None
        self.validate_data_frame = None
        self.crossEntropyLoss = None

        self.convolutional_layer1 = convolutional_layer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convolutional_layer2 = convolutional_layer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convolutional_layer3 = convolutional_layer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.linear_layers = linear_layers = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2)
        )

        for seq in [convolutional_layer1, convolutional_layer2, convolutional_layer3, linear_layers]:
            for layer in seq.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor):
        out = self.convolutional_layer1(x)
        out = self.convolutional_layer2(out)
        out = self.convolutional_layer3(out)
        out = out.view(-1, 2048)
        out = self.linear_layers(out)

        return out

    # ------Preparing the data------
    # calculate the weight vector
    # prepare the train/test dataset

    def prepare_data(self) -> None:
        self.mask_data_frame = mask_data_frame = pd.read_pickle(self.mask_path)
        train, validate = train_test_split(mask_data_frame, test_size=0.3, random_state=0,
                                           stratify=mask_data_frame["mask"])
        self.train_data_frame = MaskSet(train)
        self.validate_data_frame = MaskSet(validate)

        # penalize the network more for mistakes because of the small dataset => assign more weight
        mask_num = mask_data_frame[mask_data_frame["mask"] == 1].shape[0]
        non_mask_num = mask_data_frame[mask_data_frame["mask"] == 0].shape[0]
        samples = [non_mask_num, mask_num]
        normed_weights = [1 - (x / sum(samples)) for x in samples]
        self.crossEntropyLoss = CrossEntropyLoss(weight=torch.tensor(normed_weights))

    # train the model using a batch of size 32
    # multi-process data loading using 4 workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data_frame, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validate_data_frame, batch_size=32, num_workers=4)

    # we fix the learning rate to 0.00001
    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=0.00001)

    # ------Training step------
    # receive batch of samples, pass through our model -> follow() => compute the loss of batch
    # create log files through PyTorch Lightning
    def training_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:
        inputs, labels = batch["image"], batch["mask"]
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)
        tensor_board_loss = {"train_loss": loss}

        return {"loss": loss, "log": tensor_board_loss}

    # computer the accuracy and the loss
    def validation_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:
        inputs, labels = batch["image"], batch["mask"]
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)

        _, outputs = torch.max(outputs, dim=1)
        val_accuracy = accuracy_score(outputs.cpu(), labels.cpu())
        val_accuracy = torch.tensor(val_accuracy)

        return {"val_loss": loss, "val_acc": val_accuracy}

    # all data returned from validation_step() and calculate the avg accuracy and loss
    # => visualize in TensorBoard
    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["val_acc"] for x in outputs]).mean()
        tensor_board_logs = {"val_loss": avg_loss, "val_acc": avg_accuracy}
        return {"val_loss": avg_loss, "log": tensor_board_logs}


# ----------Training the model----------
if __name__ == '__main__':
    model = DetectorTrainer(Path("data/mask_dataframe.pickle"))

    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/weights.ckpt",
        save_weights_only=True,
        verbose=True,
        monitor="val_acc",
        mode="max"
    )
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=10,
                      checkpoint_callback=checkpoint_callback,
                      profiler=True
                      )
    trainer.fit(model)
