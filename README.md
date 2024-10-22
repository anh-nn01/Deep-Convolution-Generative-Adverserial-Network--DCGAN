# Generative-Adverserial-Network--DCGAN
PyTorch implementation of the DCGAN algorithm from scratch.

**Reference: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf**

# Hyper-parameters
As suggested by the paper in the reference, here are the values of the hyper-parameters to train the GAN model:</br>
* Batch size: **128**
* Input and Output image size: **64x64**
* Dimension of the input noise: **100x1x1**
* Learning rate: **0.0002**
* Momentum: [β1, β2] = **[0.5, 0.999]**

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

<br>
**Architecture:** <br>
<img src="Img/Architecture.jpg">

# Training scheme
* **Loss function**
1) Discriminator:<br>
The optimal Discriminator is:<br>
<img src="Img/DiscLoss.jpg"><br>
Where **x** is the training data (real), **z** is the noise, **G(z)** is the generated data from noise **z** (fake), and **D** is the Discriminator.<br>
        - Explanation:<br>
        The Discriminator is simply a Binary Classifier with Convolutional Layers and Dense Layers. Its objective is to classify training data as "real" data and generated data (outputs from the Generator) as "fake". To do this, we use a log likelihood loss function to measure how far the label of "real" data from 1.0 and how far the label of "fake" data from 0.0. Normally, we should minimize this loss function. However, in the formula above, we take **argmax** because log loss functions are usually written with a negative sign before the log function. Minimizing the negation of such loss function is equivalent to maximizing the above loss function (without negative sign).

2) Generator<br>
The optimal Generator is:<br>
<img src="Img/GenLoss.jpg"><br>
Where **z** is the noise, **G(z)** is the generated data from noise **z**, and **D** is the Discriminator.<br>
        - Explanation:<br>
        The loss fucntion of the Generator has the same idea with that of the Discriminator. The difference is that instead of trying to classify "fake" and "real" images, the Generator learns to generate "fake" images as realistic as possible. Therefore, one way to measure and optimize this realism is to use the Discriminator: while training the Generator, we want the Discriminator's outputs for the "fake" images to be close to 1.0, instead of 0.0. In other words, to train the Generator, we label generated images as 1.0 ("real"). The gradient descent optimization for the Generator works exactly as it does for the Discriminator.

* Training scheme:<br>
    - First, the Discriminator was trained on the dataset to classify those data as "real". Next, the Generator will take a batch of random noises and outputs generated images. Those generated images were used to trained the Discriminator to regconize "fake" images. By training the Discriminator on both "real" and "fake' images, the Discriminator not only captures the data distribution of real faces but also that of generated faces and learns to distinguish two data distributions. While the Discriminator trains its weights, the weights of the Generator becomes independent parameters (frozen)<br>
    - Second, the Generator was trained so that the Discriminator classifies its outputs as "real". From the first step, the Discriminator has learned to capture the data distribution of real images, thus the Generator can use the Discriminator in its loss function and exploit the Discriminator's knowledge to optimize the data distribution of generated images so that it converges to that of real images.
    - We repeat step 1 and step 2 until the data distribution of generated images converges to that of real images. At this point, the Generator can generate realistic images and the Discriminator can no longer exploit the distribution mismatch between the dataset and the generated data to classify generated data as "fake".
    
    <img src="Img/Distribution.jpg"><br>
    **The graph above was taken from the GAN paper in the reference**<br>
                        - Dotted black curve represents dataset distribution (or "real" distribution/ desirable target distribution) - Distribution of **x**<br>
                        - Green curve represents current output distribution of the Generator - Distribution of **G(z)**<br>
                        - Dotted Blue curve represents the Discriminator. This curve must divide the real distribution and the generated distribution as much as possible.<br>
                        - After each training epoch, the distribution of **G(z)** gradually converges to that of **x**, and the Discriminator can no longer exploit the difference in distribution to classify the two classes correctly.


* The Generator tries to learn and capture distribution of human faces from Celeb A Dataset and then maps the noise z to an image with similar data distribution to ouput realistic faces. When both the Generator and the Discriminator are trained, they eventually converges to the optimal point where the Generator can generate realistic outputs. The intuition is that when there is still room for improvement, the discriminator can still exploit the data distribution mismatches between Generator's ouputs and training dataset to distinguish "real" and "fake" data. When the Generator becomes better, however, the data distribution of the Generated outputs converges to that of real data, hence the Discriminator can no longer exploit the data distribution mismatches to correctly classify "real" and "fake" images. Intuitively, the generated images are too realistic to be classified as "fake", hence the Generator is successfully trained to generate realistic faces.

* In practice, the Generator and the Discriminator might never converge to such equilibirium state; therefore, the outputs may not perfectly realistic to human eyes but only have roughly similar distribtion with acceptable level of realism.

# Core Idea of GAN
* In machine learning, although the algorithms can automatically optimize and update the parameters to capture the distribution of the dataset, it is still human's work to specify the optimization objective of the algorithm. Specifically, we still need to hand-engineer the loss function so that machine learning algorithm can optimize on that loss function (in other words, loss function is a mean to communicate with a learning algorithm). 
* Different applications requires different loss function; for instance, the loss function in Neural Style Transfer uses Gram matrices as a mean to estimate the "style" of an image. Many application requires extremely complicated loss function, some of them are even considered intractable. For example, it is almost impossible to design a loss function for a regular Convolutional Decoder to map a random noise to a realistic human face.
* The core idea of GAN is that it can automatically learn the Loss function for us! The Loss function is a function of the Discriminator, and the Discriminator is technically a set of trainable parameters. As a result, our loss function is trainable, and it is trained until it cannot be improved any further, which mean D converges to an optimal point.
* Loss function: **L(G)=log(D(G(z)))**, where G* = argmax log(D(G(z))
* **L(G)** can be learned automatically without having to be explicitly defined in non-GAN learning algorithms.

# Result
* Notable Outputs:<br>
<img src="Img/1.jpg"><br>
<img src="Img/2.jpg"><br>
<img src="Img/3.jpg"><br>
<img src="Img/4.jpg"><br>
<img src="Img/5.jpg"><br>
<img src="Img/6.jpg"><br>
<img src="Img/7.jpg"><br>
<img src="Img/8.jpg"><br>
<img src="Img/9.jpg"><br>
<img src="Img/10.jpg"><br><br><br>
* What is good and what can be further improved:
1. Good:<br>
    - The Generator is successfull in capturing overall structure of human faces.<br>
    - Some outputs have highly realistic details.<br>
    - The Generator maps completely random, non-sense noise to a regconizable human faces, which is impressive as facial feature distribution are very hard to capture to regenerate such distribution.<br>
    - The outputs have high variance in emotion, facial features (such as eyes, hair, hair color, face shape, etc), and angle from the camera.

2. What can be improved:<br>
    - Some outputs are not realistic, or even scarry (you can load my Generator's weights and try it yourself) :)) My Generator was trained for 100 epochs. You can try training the model for longer to see whether there are any improvements in the output quality. <br>
    - Low resolution. You can train the Generator and Discriminator on inputs with higher resolution. Make sure you have access to sufficiently powerful GPUs to do this.<br>
    - Most of the outputted faces resemble white people's features. This is due to the fact that Celeb A dataset contains mostly images of white celebrity, which causes the Generator learn to capture only the data distribution of white's people face. You can fix this by adding an equivalent number of datapoints of non-white people to the original dataset and train your GAN on this new dataset.
