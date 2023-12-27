"""Created by agarc the 27/12/2023.
Features:
"""

from simplevit_mnist.mnist_dataloader import MnistDataLoader
from simplevit_mnist.model_configurator import SimpleViTConfigurator
from simplevit_mnist.train_engine import TrainEngine

save_path = r"./simplevit_mnist/data"

# prepare dataset and dataloader
batch_size = 1024
dataset_path = save_path + "/datasets"
train_dataloader, test_dataloader = MnistDataLoader.get_dataloaders(batch_size=batch_size,
                                                                    save_path=dataset_path)

# generate model, optimizer and loss function
model, optimizer, loss_fn, device = SimpleViTConfigurator.get_vit_config(batch_size=1024,
                                                                         image_size=(1, 28, 28),
                                                                         channels=1,
                                                                         patch_size=7,
                                                                         num_classes=10,
                                                                         dim=256,
                                                                         depth=6,
                                                                         heads=8,
                                                                         mlp_dim=512,
                                                                         lr=1e-4
                                                                         )

# Train the model and save the training results to a dictionary/figures
epoch = 25
save_path = save_path + "/training_results"
results = TrainEngine.train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=epoch,
                            device=device,
                            save_figure_path=save_path,
                            verbose_rate=5
                            )
