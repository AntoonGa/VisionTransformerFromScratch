# Vision transformer and CNN implementation in Pytorch
Simple yet optimized implementations of Vision Transformer models in Pytorch.
To use this repo just run any script in the "jobs" folder.

The wrapper class is used as a Keras/Lightning implementation without the hassle.
I've taken care of optimizing training and loading such as to get maximum duty cycle on the GPU.
- For faster IO use the HDF5 converter which will increase image loading by about 50%
- A custom Pytorch Dataloader is available to avoid the famous lag at each epoch

The code is abstract enough such as to accommodate any image datasets.

Many models to come, including Simplevit, EfficientNet, and more.


## Models
### 1. Simplevit:
A really smart transformer model than fixes (almost) all the issues with the original ViT model.
- Runs on a 3Go GPU with MNIST BatchSize = 512.
- Reaches 98% acc on MNIST after 10 epochs (1 minutes on a 3070ti/5600x).
- 60% acc on PlayingCard dataset (small dataset)
- Reaches 75% acc with minor data augmentation (best I've seen w/o pretraining & external datasets).

![GridSample - Epoch 25.png](assets%2FGridSample%20-%20Epoch%2025.png)

![confusion - Epoch 25.png](assets%2Fconfusion%20-%20Epoch%2025.png)

![playingCarsSimpleVit.png](assets%2FplayingCarsSimpleVit.png)

### 2. Efficientnetb0
An efficient CNN model that reaches SOTA on many image datasets and small local machines.
Boring but useful to set benchmarks nonetheless.
- Particularly efficient on small datasets (use pretrained!).
- Reaches 99% acc on MNIST after 1 epoch
- Reaches 99% acc on PlayingCard dataset (small dataset)


![effi_mnist_learningcurve.png](assets%2Feffi_mnist_learningcurve.png)

## References:
SimpleVit:
- https://arxiv.org/abs/2205.01580
- https://github.com/lucidrains/vit-pytorch

EfficientNet:
- https://arxiv.org/abs/1905.11946

Multiepoch Dataloader:
- https://github.com/yoniaflalo


## TODO:
- Add signal a few signal-processing-themed datasets (NMR, MRI, DAS etc...).
- MLflow integration.
- Implement a CNN-Simplevit hybrid model on high resolution images.
- Add unsupervised learning models.

## DONE:
- Abstract away model generation
- Abstract away training in a Keras-like fashion
- Implement a custom Pytorch Dataloader to avoid lag between epochs
- Implement an HDF5 converter and dataloader to speed up IO
