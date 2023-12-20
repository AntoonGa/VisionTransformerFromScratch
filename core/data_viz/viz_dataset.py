"""Created by agarc the 19/12/2023.
Features:
"""

import matplotlib.pyplot as plt
import torch


class DataViz:

    @staticmethod
    def show_image_grid(dataset: torch.Tensor) -> None:  # noqa: ANN401
        num_rows = 4
        num_cols = 6

        # Create a figure with subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        # Iterate over the subplots and display random images from the training dataset
        img_idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                # Display the image in the subplot
                image = dataset[img_idx]
                axs[i, j].imshow(image.permute(1, 2, 0))
                # Disable the axis for better visualization
                axs[i, j].axis(False)
                # Increment the iteration variable
                img_idx += 1

        # Set the super title of the figure
        fig.suptitle(f"Random {num_rows * num_cols} images from the training dataset", fontsize=16,
                     color="white")

        # Set the background color of the figure as black
        fig.set_facecolor(color="black")

        # Display the plot
        plt.show()

    @staticmethod
    def display_single_image(image: torch.Tensor) -> None:
        # Create a new figure
        fig = plt.figure()
        # Display the random image
        plt.imshow(image.permute((1, 2, 0)))
        # Disable the axis for better visualization
        plt.axis(False)
        # Set the background color of the figure as black
        fig.set_facecolor(color="black")
        plt.show()
