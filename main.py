import cv2, utils
import decompose
import sys


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    scale = 0.15
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgY = imgYCC[:, :, 0]
    M, Ds = decompose.edge_preserving_decompose(imgY, 2)
    imgYCC[:, :, 0] = M
    imgBGR = cv2.cvtColor(imgYCC, cv2.COLOR_YCR_CB2BGR)
    utils.show_image(imgBGR)
    cv2.imwrite('de_street.jpg', imgBGR)

