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

from core.classifier_models.simplevit import SimpleVitClassifier
from core.dataloaders.images_dataset import ImageDataset
from core.wrappers.wrapper import Wrapper
from shared_modules.display_engine import DisplayImages, DisplayMetrics

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_dir = r"./datasets/playing_cards/train"
valid_data_dir = r"./datasets/playing_cards/valid"
test_data_dir = r"./datasets/playing_cards/test"

num_epochs = 10
size = (128, 128)
batch_size = 32 * 4
preload = False
if preload:
    workers = 0
    prefetch = None
    persistent_workers = False
else:
    workers = 6
    prefetch = workers * 8
    persistent_workers = True

# %%
if __name__ == "__main__":
    # setting up transformation
    transformation = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    # creating dataset
    train_dataset = ImageDataset(
        data_dir=train_data_dir,
        transform=transformation,
        preload=preload
    )
    valid_dataset = ImageDataset(
        data_dir=valid_data_dir,
        transform=transformation,
        preload=preload
    )
    # generating dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  prefetch_factor=prefetch,
                                  persistent_workers=persistent_workers,
                                  pin_memory=False)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=workers,
                                  prefetch_factor=prefetch,
                                  persistent_workers=persistent_workers,
                                  pin_memory=False)

    # model
    image_size = (3, size[0], size[1])
    model = SimpleVitClassifier(image_size=image_size,
                                patch_size=16,
                                channels=3,
                                num_classes=53,
                                depth=4,
                                heads=6,
                                mlp_dim=256)
    # create optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # create wrapper
    classifier = Wrapper(model=model, optimizer=optimizer, criterion=criterion, device=device)
    summary(model=classifier.model,
            input_size=(batch_size, image_size[0], image_size[1], image_size[2]),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    # train
    classifier.fit(epochs=num_epochs,
                   train_dataloader=train_dataloader,
                   valid_dataloader=valid_dataloader)
    DisplayMetrics.display_metrics(classifier.train_losses,
                                   classifier.val_losses,
                                   classifier.train_acc,
                                   classifier.val_acc)

    # test model
    test_dataset = ImageDataset(data_dir=test_data_dir,
                                transform=transformation,
                                preload=False
                                )
    # generating dataloaders
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=2,
                                 prefetch_factor=4,
                                 persistent_workers=False,
                                 pin_memory=False)

    # model accuracy post training
    loss, acc = classifier.evaluate(test_dataloader)
    print("model loss and accuracy", loss, acc)
    for _, (image, label) in enumerate(test_dataloader):
        class_pred = classifier.predict_class(image.to(device))
        DisplayImages.display_image(image, label, class_pred, 6, 5)
        break

    labels = []
    class_preds = []
    for _, (image, label) in enumerate(test_dataloader):
        class_pred = classifier.predict_class(image.to(device))
        labels.extend(label.numpy())
        class_preds.extend(class_pred)

    DisplayMetrics.confusion_matrix(labels, class_preds, (20, 20))
    DisplayMetrics.prediction_distribution(labels, class_preds, (20, 20))
