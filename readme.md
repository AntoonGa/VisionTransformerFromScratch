# Vision transformer and CNN implementation in Pytorch
Many models to come, including Simplevit, EfficientNet, and more.
The code is abstract enough such as to accomodate any image datasets.
The wrapper class is used as a Keras Implementation.
Each model/dataset are documented in a J.notebook.

# Models
### 1. Simplevit_mnist:
- Simplevit model for mnist dataset.
- Runs on a 3Go GPU with BS 512.
- Reaches 98% acc on MNIST after 10 epochs (1 minutes on a 3070ti/5600x).
- 60% acc on PlayingCard dataset (small dataset)
-
![GridSample - Epoch 25.png](assets%2FGridSample%20-%20Epoch%2025.png)

![confusion - Epoch 25.png](assets%2Fconfusion%20-%20Epoch%2025.png)

### 2. Efficientnetb0
- An efficient CNN model
- Particularly efficient on small datasets (use pretrained!)
- Reaches 99% acc on MNIST after 1 epoch

### TODO:
- Adding profiler to the training loop.
- Target signal processing-themed datasets (NMR, MRI, DAS etc...).
- MLflow integration.
- Implement a CNN-Simplevit hybrid model on high resolution images.
- Add unsupervised learning to the training loop.

### References:
SimpleVit: https://arxiv.org/abs/2205.01580
EfficientNet: https://arxiv.org/abs/1905.11946


https://github.com/lucidrains/vit-pytorch
