# mnist.py
# Functions for working with the MNIST dataset.
#
# For details about the dataset, check out the following link:
# http://yann.lecun.com/exdb/mnist/

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage


def load_images(filename):
    with open(filename, 'rb') as fid:
        # Read magic number
        magic = np.fromfile(fid, '>i4', 1)
        assert magic[0] == 2051, "Bad magic number in {} (expected 2051, but got {})".format(filename, magic[0])

        # Read number and size of images
        num_images = np.fromfile(fid, '>i4', 1)
        num_rows = np.fromfile(fid, '>i4', 1)
        num_cols = np.fromfile(fid, '>i4', 1)

        # Read image data
        images = np.fromfile(fid, '>u1').reshape((num_images[0], num_rows[0], num_cols[0])).transpose((1, 2, 0))
        return images


def load_labels(filename):
    with open(filename, 'rb') as fid:
        # Read magic number
        magic = np.fromfile(fid, '>i4', 1)
        assert magic[0] == 2049, "Bad magic number in {} (expected 2049, but got {})".format(filename, magic[0])

        # Read number and size of images
        num_images = np.fromfile(fid, '>i4', 1)

        # Read image data
        labels = np.fromfile(fid, '>u1').reshape((num_images[0], -1))
        return labels


def show_images(images, N=1, shape=None):
    # Show N random samples from the dataset.
    ind = np.random.choice(images.shape[2], N, replace=False)
    ind.shape = (len(ind),)

    if shape is None:
        s = int(np.ceil(N**(0.5)))
        shape = (s, s)
    m = montage(images[:, :, ind].transpose(2, 0, 1), grid_shape=shape)
    plt.imshow(m, cmap='gray')
    plt.axis('off')
    plt.show()
