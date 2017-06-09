import cv2, matplotlib.pyplot as plt, numpy as np
import os
from math import ceil



def plot_hist(xs, range, bins=20):
    plt.hist(xs, bins, range=range)
    plt.show()

def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])



def show_image(cv_image, cmap='gray'):
    if cv_image.ndim > 2:
        plt.imshow(bgr_to_rgb(cv_image))
    else:
        plt.imshow(cv_image, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_images(cv_images, param_txts=None, cols=2, cmap='gray'):
    rows = int(ceil(len(cv_images) / cols))
    for i, cv_image in enumerate(cv_images):
        plt.subplot(rows, cols, i + 1)
        if cv_image.ndim > 2:
            plt.imshow(bgr_to_rgb(cv_image))
        else:
            plt.imshow(cv_image, cmap=cmap)

        if param_txts:
            plt.title(param_txts[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    if param_txts:
        plt.subplots_adjust(left=0.01, right=0.99, bottom= 0.01, top=0.95, wspace=0.02, hspace=0.2)
    else:
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)

    plt.show()

