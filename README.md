# Generative-Adverserial-Network--DCGAN
PyTorch implementation of the original GAN algorithm from scratch.

**Reference: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf**

# Hyper-parameters
As suggested by the paper in the reference, here are the values of the hyper-parameters to train the GAN model:</br>
* Batch size: **128**
* Input and Output image size: **64x64**
* Dimension of the input noise: **100x1x1**
* Learning rate: **0.0002**
* Momentum: [beta1, beta2] = **[0.5, 0.999]**

# Dataset
* The GAN model was trained and tested on CelebA dataset. The Generator was trained to capture the data distribution of human face on CelebA dataset and tried to generate new human faces as realistic as possible.
* You can download the dataset to your local machine [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg).

# Generator Architecture
* The Generator Architecture is the Decoder part in Encoder-Decoder architecture. The Generator was trained to map a random noise with size **(100x1x1)** to an image of human face with size **(3x64x64)**.
* The architecture:

(Transpose Convolution - BatchNorm - ReLU) * 4 - (Transpose Convolution - Tanh)

# Discriminator Architecture
* The Discriminator Architecture is a Binary Convolutional Classifier, which was trained to classified outputs from the Generator as "fake" (labeled 0) and the real images from dataset as "real" (labeled 1). The expected input for the Discriminator Convolutional Neural Network is a batch of images with size **(3x64x64)** 

* The architecture:

(Convolution - LeakyReLU) - (Convolution - BatchNorm - LeakyReLU) * 3 - (Convolution - Sigmoid)

# Training scheme
* Loss function (explain log)

* Learn and capture distribution of human faces from Celeb A Dataset and try to map noise z to an image with similar data distribution to ouput realistic faces.

* When there is still room for improvement, the discriminator can still exploit the data distribution mismatches between Generator's ouputs and training dataset to distinguish "real" and "fake" data. When the Generator becomes better, however, the data distribution of the Generated outputs converges to that of real data, hence the Discriminator can no longer exploit the data distribution mismatches to correctly classify "real" and "fake" images. Intuitively, the generated images are too realistic to be classified as "fake", hence the Generator is successfully trained to generate realistic faces.

* In practice, the Generator and the Discriminator might never converge to such equilibirium state; therefore, the ouptuts may not perfectly realistic to human eyes but only have roughly similar distribtion with acceptable level of realism.

# Core Idea of GAN
* In machine learning, although the algorithms can automatically optimize and update the parameters to capture the distribution of the dataset, it is still human's work to specify the optimization objective of the algorithm. Specifically, we still need to hand-engineer the loss function so that machine learning algorithm can optimize on that loss function (in other words, loss function is a mean to communicate with a learning algorithm). 
* Different applications requires different loss function; for instance, the loss function in Neural Style Transfer uses Gram matrices as a mean to estimate the "style" of an image. Many application requires extremely complicated loss function, some of them are even considered intractable. For example, it is almost impossible to design a loss function for a regular Convolutional Decoder to map a random noise to a realistic human face.
* The core idea of GAN is that it can automatically learn the Loss function for us! The Loss function is a function of the Discriminator, and the Discriminator is technically a set of trainable parameters. As a result, our loss function is trainable, and it is trained until it cannot be improved any further, which mean D converges to an optimal point.
* Loss function: **L(G)=log(D(G(z)))**, where G* = argmax log(D(G(z))
* **L(G)** can be learned automatically without having to be explicitly defined in non-GAN learning algorithms.
