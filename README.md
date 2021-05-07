# PyTorch-2D-3D-UNet-Tutorial

This repository contains all files that were used for the blog series 'Creating and training a U-Net model with PyTorch for 2D & 3D semantic segmentation - A guide to semantic segmentation with PyTorch and the U-Net'.
You can find the blog posts [here](https://johschmidt42.medium.com/).

In [requirements.txt](requirements.txt) you'll find the packages for the conda environment that I used. This does not necessarily mean that these will work on your machine/computer (for example torch comes with a specific cuda version).

I have updated the repo, e.g. the [transformations](transformations.py) and added a dataset example for a [3D dataset](Part5-3D-example.ipynb).
There is also an [example](Part6-PL-example.ipynb) now that shows how you can use a segmentation model like the UNet (or any other segmentation model) in PyTorch Lightning in combination with a logger like [neptune.ai](https://neptune.ai/) for experiment tracking.

Therefore, the blog series needs to be updated. Stay tuned. 
