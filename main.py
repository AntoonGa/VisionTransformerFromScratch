"""Created by agarc the 19/12/2023.
Features:
"""
from core.data_fetcher.data_preparer import DataPreparer
from core.data_viz.viz_dataset import DataViz
from core.data_worker.data_worker import DataWorker

# from torchinfo import summary
#
# from core.data_fetcher.data_preparer import DataPreparer
# from transformer.vit_assembly.vit_assembly import ViT
#
# BATCH_SIZE = 32
# vit = ViT()
#
# summary(model=vit,
#         input_size=(BATCH_SIZE, 3, 224, 224),  # (batch_size, num_patches, embedding_dimension)
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])
#
# data_prepare = DataPreparer()
# data = data_prepare.create_transformed_dataloader("test")
# batch = next(iter(data))
# random_images, random_labels = batch
# random_images, random_labels = random_images.to("cuda"), random_labels.to("cuda")
# classes = vit(random_images)
# print("output_size", classes.shape)  # noqa: T201

data_source = "test"
data_worker = DataWorker(data_source, "cpu")
for _ in range(2):
    random_images, random_labels = next(data_worker)

    DataViz.show_image_grid(random_images)
    DataViz.display_single_image(random_images[0])
