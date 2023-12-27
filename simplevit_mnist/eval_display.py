"""Created by agarc the 27/12/2023.
Features:
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


class DisplayEngine:
    """ Computes and displays the predictions of a model on a batch of images"""

    def __init__(self, save_path : str = r"./last_training_results") -> None:
        self.save_path = save_path
        self.epoch = 0

    def plot_all(self,
                 results: dict,
                 model: torch.nn.Module,
                 data_loader: DataLoader) -> None:
        """ Run all plots and computation."""

        images, labels, pred_labels, accuracy = self._get_predictions(model, data_loader)
        self._display_grid_predictions(images, labels, pred_labels)
        self._prediction_distribution(labels, pred_labels, accuracy)
        self._predictions_confusion_matrix(labels, pred_labels)
        self._plot_training_curves(results)

    def _get_predictions(self,
                         model: torch.nn.Module,
                         data_loader: DataLoader
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """ Generate ALL predictions required to perform plots."""
        # Get a batch of images
        images, labels = next(iter(data_loader))
        # Move the batch to the device
        images = images.to("cuda")
        # Get predictions
        preds = model(images)
        # Move predictions and labels to CPU
        preds = preds.cpu()
        labels = labels.cpu()
        pred_labels = torch.argmax(preds, dim=1)
        # compute accuracy
        accuracy = torch.sum(pred_labels == labels).item() / len(labels)
        return images, labels, pred_labels, accuracy

    def _display_grid_predictions(self,
                                  images: torch.Tensor,
                                  labels: torch.Tensor,
                                  pred_labels: torch.Tensor) -> None:
        """ Display a grid of samples/prediction/labels """
        grid_x = 32
        grid_y = 32
        fig_size = (grid_y * 2, grid_x * 2)

        # Visualize the dataset in a grid
        fig, axes = plt.subplots(grid_x, grid_y, figsize=fig_size)
        for i, ax in enumerate(axes.flat):
            img = images[i].permute((1, 2, 0)).cpu().numpy()
            if labels[i] == pred_labels[i]:
                ax.imshow(img, cmap="Grays")
            else:
                ax.imshow(img)
            ax.set_title(f"L: {labels[i]} | P: {pred_labels[i]}")
            ax.axis("off")

        # Save the figure
        save_path = f"{self.save_path}/GridSample - Epoch {self.epoch}.png"
        plt.savefig(save_path)

        # Close the figure to prevent overlapping in the next iteration
        plt.close(fig)

    def _plot_training_curves(self, results: dict) -> None:
        """ Display losses and metrics vs epochs during training """
        if results:
            # Plot the training results in subplots
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            ax[0].plot(results["train_loss"], label="train loss")
            ax[0].plot(results["test_loss"], label="test loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].legend()
            ax[1].plot(results["train_acc"], label="train accuracy")
            ax[1].plot(results["test_acc"], label="test accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].legend()
            save_path = f"{self.save_path}/curves.png"
            plt.savefig(save_path)
            plt.close(fig)

    def _prediction_distribution(self,
                                 labels: torch.Tensor,
                                 pred_labels: torch.Tensor,
                                 accuracy: float) -> None:
        """ Dataset-labels and inference distribution and class histogram"""
        # Plot histogram of predictions and labels using plt.bar
        classes = list(range(10))
        unique_labels = torch.unique(labels)

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot labels
        label_counts = [torch.sum(labels == c).item() for c in classes]
        ax.bar(unique_labels, label_counts, alpha=0.5, label="labels", align="center", width=0.4)

        # Plot predictions
        pred_counts = [torch.sum(pred_labels == c).item() for c in classes]
        ax.bar(unique_labels + 0.4, pred_counts, alpha=0.5, label="predictions", align="center",
               width=0.4)

        # Adjust x-axis ticks and labels
        ax.set_xticks(unique_labels + 0.2)
        ax.set_xticklabels(classes)

        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"Set accuracy: {accuracy:.2f}")
        save_path = f"{self.save_path}/histogram - Epoch {self.epoch}.png"
        plt.savefig(save_path)
        plt.close(fig)

    def _predictions_confusion_matrix(self,
                                      labels: torch.Tensor,
                                      pred_labels: torch.Tensor
                                      ) -> None:
        """
           Generate a confusion matrix.

           Parameters:
           - actual_labels: List or array of true labels.
           - predicted_labels: List or array of predicted labels.
           - class_names: List of class names (optional).

           Returns:
           - Confusion matrix as a NumPy array.
       """

        # Calculate confusion matrix
        cm = confusion_matrix(labels.cpu().numpy(), pred_labels.cpu().numpy())

        fig = plt.figure(figsize=(10, 10))
        # Plot the confusion matrix as a heatmap
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        classes = np.unique(labels.cpu().numpy())
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Display the values in each cell
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        save_path = f"{self.save_path}/confusion - Epoch {self.epoch}.png"
        plt.savefig(save_path)
        plt.close(fig)
