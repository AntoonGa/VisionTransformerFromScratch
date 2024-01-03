import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class DisplayImages:
    @staticmethod
    def display_image(images: torch.Tensor,
                      labels: torch.Tensor,
                      preds: torch.Tensor = None,
                      grid_x: int = 6,
                      grid_y: int = 6,
                      ) -> None:
        """ Display a grid of samples/prediction/labels """
        fig_size = (grid_y * 2, grid_x * 2)

        # Visualize the dataset in a grid
        fig, axes = plt.subplots(grid_x, grid_y, figsize=fig_size)
        for i, ax in enumerate(axes.flat):
            img = images[i].permute((1, 2, 0)).cpu().numpy()
            ax.imshow(img, cmap="YlOrRd")
            try:
                ax.set_title(f"L: {labels[i]} | P: {preds[i]}")
            except:  # noqa: E722
                ax.set_title(f"L: {labels[i]}")
            ax.axis("off")
        plt.show()


class DisplayMetrics:
    @staticmethod
    def display_metrics(train_losses: list[float],
                        val_losses: list[float],
                        train_acc: list[float],
                        val_acc: list[float],
                        fig_size: tuple[int, int] = (15, 5)
                        ) -> None:
        """ Display the metrics of the training and validation """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        ax1.plot(train_losses, label="train_loss")
        ax1.plot(val_losses, label="val_loss")
        ax1.legend()
        ax1.set_title("Loss")
        ax2.plot(train_acc, label="train_acc")
        ax2.plot(val_acc, label="val_acc")
        ax2.legend()
        ax2.set_title("Accuracy")
        plt.show()

    @staticmethod
    def confusion_matrix(labels: list[int],
                         pred_labels: list[int],
                         fig_size: tuple[int, int] = (12, 12),
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
        cm = confusion_matrix(labels, pred_labels)

        plt.figure(figsize=fig_size)
        # Plot the confusion matrix as a heatmap
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        classes = np.unique(labels)
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

        plt.show()

    @staticmethod
    def prediction_distribution(labels: list[int],
                                pred_labels: list[int],
                                fig_size: tuple[int, int] = (12, 6)
                                ) -> None:
        """ Dataset-labels and inference distribution and class histogram"""
        unique_labels, counts_labels = np.unique(labels, return_counts=True)
        number_of_classes = len(unique_labels)
        # Plot histogram of predictions and labels using plt.bar
        fig, ax1 = plt.subplots(1, 1, figsize=fig_size)
        ax1.hist(labels, bins=number_of_classes, label="labels")
        ax1.hist(pred_labels, bins=number_of_classes, label="predictions", alpha=0.5)
        ax1.set_xticks(unique_labels)
        ax1.set_xticklabels(unique_labels)
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        ax1.legend()
        # display plot
        plt.show()
