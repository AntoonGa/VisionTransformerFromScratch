"""
Author: {author} || Date: {date}
Features:
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Wrapper:
    """ wrapper pipeline for pytorch models
    This is used to train and validate models"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: str = "cuda") -> None:
        """ initialize trainer pipeline
        Args:
            model: pytorch model
            optimizer: pytorch optimizer
            criterion: pytorch loss function
            device: device to run the model on """

        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model = model.to(self.device)

        # initialize lists for storing losses at each epoch
        self.train_losses, self.val_losses = [], []
        self.train_acc, self.val_acc = [], []

    def evaluate(self, dataloader: DataLoader = None) -> tuple[float, float]:
        """ return loss and accuracy over the entire dataloader """
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        nb_samples = len(dataloader.dataset)
        with torch.no_grad():
            for x_data, labels in dataloader:
                x_data = x_data.to(self.device)
                labels = labels.to(self.device)
                running_loss, correct_predictions = self._validate_on_batch(x_data,
                                                                            labels,
                                                                            running_loss,
                                                                            correct_predictions)

        loss = running_loss / nb_samples
        acc = correct_predictions / nb_samples
        return loss, acc

    def predict(self, dataloader: DataLoader = None) -> list:
        """ return class predictions over the entire dataloader """
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())

        return all_predictions

    def fit(self,
            epochs: int = 5,
            train_dataloader: DataLoader = None,
            valid_dataloader: DataLoader = None,
            ) -> None:
        """ train model for a number of epochs
        Args:
            epochs: number of epochs to train for
            train_dataloader: dataloader for training dataset
            valid_dataloader: dataloader for validation dataset """

        # run training and validation on each epoch
        for epoch in tqdm(range(epochs), desc="Epochs:"):
            self._train_on_dataset(train_dataloader)
            self._validate_on_dataset(valid_dataloader)

            print(
                f"| Epoch {epoch + 1}/{epochs} | "
                f"T-loss: {self.train_losses[-1]:.3f} | "
                f"V-loss: {self.val_losses[-1]:.3f} |"
                f"T-acc: {self.train_acc[-1]:.2f} |"
                f"V-acc: {self.val_acc[-1]:.2f}"
            )

    def _train_on_dataset(self, train_dataloader: DataLoader) -> None:
        """ run an epoch of training on the entire dataset"""
        # training phase
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        nb_samples = len(train_dataloader.dataset)
        for x_data, labels in train_dataloader:
            x_data = x_data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            running_loss, correct_predictions = self._train_on_batch(x_data,
                                                                     labels,
                                                                     running_loss,
                                                                     correct_predictions)

        train_loss = running_loss / nb_samples
        self.train_losses.append(train_loss)

        train_acc = correct_predictions / nb_samples
        self.train_acc.append(train_acc)

    def _train_on_batch(self,
                        x_batch: torch.Tensor,
                        labels_batch: torch.Tensor,
                        running_loss: float = 0.0,
                        correct_predictions: int = 0) -> tuple[float, int]:
        """ run a single training step on a batch of data
        updates running_loss and running_accuracy and returns them """
        self.optimizer.zero_grad()
        preds = self.model(x_batch)
        loss = self.criterion(preds, labels_batch)
        loss.backward()
        self.optimizer.step()

        # update running values after scaling by batch size
        running_loss += loss.item() * labels_batch.size(0)
        _, class_preds = torch.max(preds, 1)
        correct_predictions += (class_preds == labels_batch).sum().item()

        return running_loss, correct_predictions

    def _validate_on_dataset(self, valid_dataloader: DataLoader) -> None:
        """ run validation on the entire dataset"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        nb_samples = len(valid_dataloader.dataset)
        with torch.no_grad():
            for x_data, labels in valid_dataloader:
                x_data = x_data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                running_loss, correct_predictions = self._validate_on_batch(x_data,
                                                                            labels,
                                                                            running_loss,
                                                                            correct_predictions)

        val_loss = running_loss / nb_samples
        self.val_losses.append(val_loss)

        val_acc = correct_predictions / nb_samples
        self.val_acc.append(val_acc)

    def _validate_on_batch(self,
                           x_batch: torch.Tensor,
                           labels_batch: torch.Tensor,
                           running_loss: float = 0.0,
                           correct_predictions: int = 0) -> tuple[float, int]:
        """ run a single validation step on a batch of data
        updates running_loss and running_accuracy and returns them """
        preds = self.model(x_batch)
        loss = self.criterion(preds, labels_batch)

        # update running values after scaling by batch size
        running_loss += loss.item() * labels_batch.size(0)
        _, class_preds = torch.max(preds, 1)
        correct_predictions += (class_preds == labels_batch).sum().item()

        return running_loss, correct_predictions


# %%
if __name__ == "__main__":
    pass
