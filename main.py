import cv2, utils
import decompose
import sys


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    print(img.dtype)
    imgY = imgYCC[:, :, 0]
    M, Ds = decompose.edge_preserving_decompose(imgY , 1)
    utils.show_image(M)


