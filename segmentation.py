import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import measure
from PIL import Image
from scipy.stats import norm
matplotlib.use('Agg')


def load_image(filename):
    img = Image.open(filename).convert('1')  # load image file and covert it to binary (black and white)
    return np.array(img)


# labelling the connected area in the image.
def segmentation(image):
    return measure.label(image, background=0, connectivity=2)


def output_labelled(image):
    fig = plt.figure(figsize=(10, 10))
    fig.set_canvas(plt.gcf().canvas)
    plt.imshow(image, cmap='nipy_spectral')
    fig.savefig('labelled_image.jpg', dpi=300)


# clean the image and remove noise areas
def remove_noise(image, min_size):
    for label in range(1, image.max()):
        mask = np.zeros(image.shape)
        mask[image == label] = 1
        if mask.sum() < min_size:
            image[image == label] = 0


# measure the area of labelled area
def quantify_labelled_area(image):
    size_distribution = []
    for label in range(1, image.max()):
        mask = np.zeros(image.shape)
        mask[image == label] = 1
        size_distribution.append(mask.sum())
    return size_distribution


def plot_size_distribution(size_distribution):
    size_distribution.sort()
    fig = plt.figure(figsize=(10, 10))
    x = np.linspace(size_distribution[0], size_distribution[-1], 10)
    plt.hist(size_distribution, bins=x)
    (meanAverage, standardDeviation) = norm.fit(size_distribution)
    y = norm.pdf(x, meanAverage, standardDeviation)
    plt.plot(x, y, 'r--', linewidth=2)
    fig.savefig('Size_distribution.jpg', dpi=600)


def main():
    image = 1 - load_image('images/test.png')  # load and invert the image
    image = segmentation(image)
    output_labelled(image)
    size_distribution = quantify_labelled_area(image)
    plot_size_distribution(size_distribution)


if __name__ == '__main__':
    main()
