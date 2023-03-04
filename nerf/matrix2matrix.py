"""This module will generate a Conv2D to Conv2D neural network"""
import os

# Import torch
import torch
from torch import nn
import torch.nn.functional as F

# Import lightning
import lightning as pl

class Matrix2Matrix(pl.LightningModule):
    """This class will generate a Conv2D to Conv2D neural network"""
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 3, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(3, output_channels, kernel_size, stride, padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            output_channels, output_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("valid_loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

if __name__ == '__main__':

    # Load the data set in the current directory
    train_path = os.path.join(os.path.dirname(__file__), "data", "matrix_2_matrix_train.pt")
    valid_path = os.path.join(os.path.dirname(__file__), "data", "matrix_2_matrix_valid.pt")
    dataset = (
        pl.pytorch.core.datamodule.LightningDataModule.from_datasets(
            train_dataset=torch.load(train_path),
            val_dataset=torch.load(valid_path),
            batch_size=32,
            num_workers=2,
        )
    )

    # Create the model
    model = Matrix2Matrix(
        input_channels=1,
        output_channels=1,
        kernel_size=2,
        stride=1,
        padding=1
    )

    # Create the trainer
    trainer = pl.Trainer(max_epochs=10, accelerator='cpu')

    # Fit the model
    trainer.fit(model, dataset)
