"""
Author: {author} || Date: {date}
Features:
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary

from core.classifier_models.efficientnet_b0_pretrained import (
    SimpleCardClassifierEfNetb0,
)
from core.dataloaders.mnist_dataset import MnistDataset
from core.wrappers.wrapper import Wrapper
from shared_modules.display_engine import DisplayImages, DisplayMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_dir = r"./datasets/mnist/train"
valid_data_dir = r"./datasets/mnist/test"

size = (128, 128)
batch_size = 32 * 3
preload = True
if preload:
    workers = 0
    prefetch = None
    persistent_workers = False
else:
    workers = 6
    prefetch = workers * 8
    persistent_workers = True

if __name__ == "__main__":
    # setting up transformation
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # 1 channel to 3 channel requiered for this model
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    # creating dataset
    train_dataset = MnistDataset(
        data_dir=train_data_dir,
        train=True,
        transform=transformation,
        preload=preload
    )
    valid_dataset = MnistDataset(
        data_dir=valid_data_dir,
        train=False,
        transform=transformation,
        preload=preload
    )

    # preparing dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  prefetch_factor=prefetch,
                                  persistent_workers=persistent_workers,
                                  pin_memory=False)

    valid_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  prefetch_factor=prefetch,
                                  persistent_workers=persistent_workers,
                                  pin_memory=False)

    # generate model
    model = SimpleCardClassifierEfNetb0(num_classes=10, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # create wrapper
    classifier = Wrapper(model=model, optimizer=optimizer, criterion=criterion, device=device)
    input_size = (batch_size, 3, size[0], size[1])
    summary(model=classifier.model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    num_epochs = 5
    classifier.fit(epochs=num_epochs, train_dataloader=train_dataloader,
                   valid_dataloader=valid_dataloader)

    # model accuracy post training
    loss, acc = classifier.evaluate(valid_dataloader)
    print("model loss and accuracy", loss, acc)
    for _, (image, label) in enumerate(valid_dataloader):
        class_pred = classifier.predict_class(image.to(device))
        DisplayImages.display_image(image, label, class_pred, 6, 5)
        break

    DisplayMetrics.display_metrics(classifier.train_losses,
                                   classifier.val_losses,
                                   classifier.train_acc,
                                   classifier.val_acc)

    labels = []
    class_preds = []
    for _, (image, label) in enumerate(valid_dataloader):
        class_pred = classifier.predict_class(image.to(device))
        labels.extend(label.numpy())
        class_preds.extend(class_pred)

    DisplayMetrics.confusion_matrix(labels, class_preds, (20, 20))
    DisplayMetrics.prediction_distribution(labels, class_preds, (20, 20))
