import cv2, utils
import decompose
import sys


if __name__ == '__main__':
    filename = sys.argv[1]
    img = cv2.imread(filename)
    scale = 0.13
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgY = imgYCC[:, :, 0]
    M, Ds = decompose.edge_preserving_decompose(imgY, 2)
    M = decompose.adjust_constrast(imgY, M)
    imgYCC[:, :, 0] = M
    imgBGR = cv2.cvtColor(imgYCC, cv2.COLOR_YCR_CB2BGR)
    utils.show_image(imgBGR)
    cv2.imwrite('de_' + filename, imgBGR)

