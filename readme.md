# Vision transformer implementation in Pytorch

## Each subfolder is a different model applied to a different dataset.

### Simplevit_mnist:
- Simplevit model for mnist dataset.
- Runs on a 3Go GPU with BS 512.
- Reaches 95% acc after 10 epochs (3 minutes on a 3070ti/5600x).
- Simply run ./simplevit_mnist/main.py to instantiate the model and train it.

![GridSample - Epoch 25.png](simplevit_mnist%2Fdatasets%2FGridSample%20-%20Epoch%2025.png)

![confusion - Epoch 25.png](simplevit_mnist%2Fdatasets%2Fconfusion%20-%20Epoch%2025.png)

### TODO:
- target signal processing-themed datasets (NMR, MRI, DAS etc...).
- MLflow integration.
- Implement a CNN-Simplevit hybrid model on high resolution images.

### References:
https://arxiv.org/abs/2205.01580

https://github.com/lucidrains/vit-pytorch
