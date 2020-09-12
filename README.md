# Generative-Adverserial-Network--DCGAN
PyTorch implementation of the original GAN algorithm from scratch.

**Reference: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf**

# Hyper-parameters:
As suggested by the paper in the reference, here are the values of the hyper-parameters to train the GAN model:</br>
* Batch size: **128**
* Input and Output image size: **64x64**
* Dimension of the input noise: **100x1x1**
* Learning rate: **0.0002**
* Momentum: [beta1, beta2] = [0.5, 0.999]

# Dataset:
* The GAN model was trained and tested on CelebA dataset. The Generator was trained to capture the data distribution of human face on CelebA dataset and tried to generate new human faces as realistic as possible.
* You can download the dataset to your local machine [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg).

