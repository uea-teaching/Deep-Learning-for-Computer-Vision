# Deep Learning for Computer Vision

Practical examples of deep learning for computer vision - using **pytorch**.

## Requirements

The examples require a `pytorch` installation which is best achieved using a **conda** environment.

I have provided an environment file, torch-cv-env.yml which can be used to install the dependencies.

    conda env update -f torch-cv-env.yml

To obtain a working `conda` environment, you will need visit the Anaconda
[website](https://www.anaconda.com/products/individual),
and download the installer for your platform.

**NOTE**: for Mac users, `cuda` is not supported,
I have provided an alternate environment file, `torch-cv-env-mac.yml`, just for this platform.

Unfortunately, you will have reduced performance without `cuda`.

To activate this environment, use

    $ conda activate torch-cv

To deactivate an active environment, use

    $ conda deactivate

## Data

We will work with the following datasets:

### MNIST

MNIST (Modified National Institute of Standards and Technology) handwritten digits.
More information can be found [here](http://yann.lecun.com/exdb/mnist/).

Greyscale images of handwritten digits 28 x 28 pixels.

The MNIST database contains 60,000 training images and 10,000 testing images.

### CIFAR10

The CIFAR10 (Canadian Institute For Advanced Research) database contains 50,000 training images and 10,000 testing images.

More information [here](https://www.cs.toronto.edu/~kriz/cifar.html).
These are 32 x 32 colour images of various objects in 10 classes.

## Transfer learning

We will experiment with transfer learning using the following models:

VGG11 - for fine tuning the whole model.
ResNet18 - as a feature extractor.
