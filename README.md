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
