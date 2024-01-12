"""
Author: {author} || Date: {date}
Features:
    - SimpleVitClassifier on playing cards dataset
    - the model does not go over 50% accuracy and overfits, so augmentation is needed
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transformsv2
from torchinfo import summary

from core.classifier_models.simplevit import SimpleVitClassifier
from core.wrappers.dataloader_wrapper import dataloader_wrapper
from core.wrappers.model_wrapper import Wrapper
from shared_modules.display_engine import DisplayMetrics

###########################
# CUDA params
###########################
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################
# Training params
###########################
num_epochs = 100
batch_size = 128
lr = 1e-4

###########################
# Data params
###########################
data_dir = r"./datasets/birds"
dataset_type = "image"
dataloader_type = "multiepoch"
data_file = "data_128x128.hdf5"
preload = False
nb_classes = 525
target_resize = (128, 128)
channels = 3


# data_dir = r"./datasets/playing_cards"
# dataset_type = "hdf5"
# dataloader_type = "multiepoch"
# data_file = "data_128x128.hdf5"
# preload = False
# nb_classes = 53
# target_resize = (128, 128)
# channels = 3

###########################
# Model
###########################
image_size = (channels, target_resize[0], target_resize[1])
input_size = (batch_size, image_size[0], image_size[1], image_size[2])

model = SimpleVitClassifier(image_size=image_size,
                            patch_size=16,
                            channels=3,
                            num_classes=nb_classes,
                            depth=4,
                            heads=6,
                            mlp_dim=256)

###########################
# Transformation & Augmentation
###########################
# do-nothing
transformation = transformsv2.Compose([
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    transformsv2.Resize(target_resize, antialias=True),
])

# augmentation
augmentation = transformsv2.Compose([
    transformsv2.ToImage(),
    transformsv2.ToDtype(torch.float32, scale=True),
    transformsv2.Resize(target_resize, antialias=True),
    # we can use random choice to apply random transformations, identify when using raw image
    # flip/rotations transforms
    transformsv2.RandomChoice([
        transformsv2.RandomVerticalFlip(p=1),
        transformsv2.RandomHorizontalFlip(p=1),
        transformsv2.RandomRotation(degrees=45),
        torch.nn.Identity(),
    ]),
    # affine transforms
    transformsv2.RandomChoice([
        transformsv2.RandomAffine(degrees=0, translate=(0., 0.2)),
        torch.nn.Identity(),
    ]),
    # color transforms
    transformsv2.RandomChoice([
        transformsv2.RandomAdjustSharpness(sharpness_factor=2),
        transformsv2.RandomEqualize(p=1),
        transformsv2.RandomPosterize(bits=2, p=1),
        transformsv2.RandomInvert(p=1),
        torch.nn.Identity(),
    ]),
    # Gaussian noise
    transformsv2.RandomChoice([
        transformsv2.GaussianBlur(kernel_size=3, sigma=(0.001, 0.1))
    ]),

])

# dataloaders parameters
train_data_params = {"dataset_type": dataset_type,
                     "data_dir": data_dir + "/train",
                     "data_file": data_file,
                     "transformation": augmentation,
                     "preload": preload,
                     "dataloader_type": dataloader_type,
                     "batch_size": batch_size,
                     "shuffle": True,
                     "num_workers": 6,
                     "prefetch_factor": 60,
                     "persistent_workers": True}

valid_data_params = {"dataset_type": dataset_type,
                     "data_dir": data_dir + "/valid",
                     "data_file": data_file,
                     "transformation": transformation,
                     "preload": False,
                     "dataloader_type": dataloader_type,
                     "batch_size": batch_size,
                     "shuffle": False,
                     "num_workers": 0,
                     "prefetch_factor": None,
                     "persistent_workers": False}

test_data_params = {"dataset_type": dataset_type,
                    "data_dir": data_dir + "/test",
                    "data_file": data_file,
                    "transformation": transformation,
                    "preload": False,
                    "dataloader_type": dataloader_type,
                    "batch_size": batch_size,
                    "shuffle": False,
                    "num_workers": 0,
                    "prefetch_factor": None,
                    "persistent_workers": False}

# %%
if __name__ == "__main__":
    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    summary(model=model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # create wrapper
    classifier = Wrapper(model=model, optimizer=optimizer, criterion=criterion, device=device)
    # train
    classifier.fit(epochs=num_epochs,
                   train_dataloader=dataloader_wrapper(**train_data_params),
                   valid_dataloader=dataloader_wrapper(**valid_data_params)
                   )

    DisplayMetrics.display_metrics(classifier.train_losses,
                                   classifier.val_losses,
                                   classifier.train_acc,
                                   classifier.val_acc)

    #############################
    # model accuracy post training
    #############################
    # from shared_modules.display_engine import DisplayImages
    # loss, acc = classifier.evaluate(dataloader_wrapper(**test_data_params))
    # print("model loss and accuracy", loss, acc)
    # for _, (image, label) in enumerate(dataloader_wrapper(**test_data_params)):
    #     class_pred = classifier.predict_class(image.to(device))
    #     DisplayImages.display_image(image, label, class_pred, 6, 5)
    #     break
    #
    # labels = []
    # class_preds = []
    # for _, (image, label) in enumerate(dataloader_wrapper(**test_data_params)):
    #     class_pred = classifier.predict_class(image.to(device))
    #     labels.extend(label.numpy())
    #     class_preds.extend(class_pred)
    #
    # DisplayMetrics.confusion_matrix(labels, class_preds, (20, 20))
    # DisplayMetrics.prediction_distribution(labels, class_preds, (20, 20))
