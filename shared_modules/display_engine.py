import matplotlib.pyplot as plt
import torch


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
            if preds:
                ax.set_title(f"L: {labels[i]} | P: {preds[i]}")
            else:
                ax.set_title(f"L: {labels[i]}")
            ax.axis("off")
        plt.show()


class DisplayMetrics:
    @staticmethod
    def display_metrics(train_losses: list[float],
                        val_losses: list[float],
                        train_acc: list[float],
                        val_acc: list[float]) -> None:
        """ Display the metrics of the training and validation """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(train_losses, label="train_loss")
        ax1.plot(val_losses, label="val_loss")
        ax1.legend()
        ax1.set_title("Loss")
        ax2.plot(train_acc, label="train_acc")
        ax2.plot(val_acc, label="val_acc")
        ax2.legend()
        ax2.set_title("Accuracy")
        plt.show()
