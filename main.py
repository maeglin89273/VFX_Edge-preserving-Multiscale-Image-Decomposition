import cv2, utils
import numpy as np
import decompose
import sys


if __name__ == '__main__':
    filename = sys.argv[1]
    img = cv2.imread(filename)
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgY = imgYCC[:, :, 0]
    M, Ds = decompose.edge_preserving_decompose(imgY, int(sys.argv[2]))
    enhanced_factor = 2

    adj_cont_M = decompose.adjust_constrast(imgY, M)
    m_show = np.copy(imgYCC)
    detail_enhanced = np.copy(imgYCC)

    m_show[:, :, 0] = adj_cont_M
    detail_enhanced[:, :, 0] = decompose.clip_and_convert_to_uint8(M + enhanced_factor * np.sum(Ds, axis=0))
    # detail_enhanced[:, :, 0] = decompose.clip_and_convert_to_uint8(M + enhanced_factor * Ds[0] + Ds[1])
    # detail_enhanced[:, :, 0] = decompose.clip_and_convert_to_uint8(M + Ds[0] + enhanced_factor * Ds[1])

    m_show = cv2.cvtColor(m_show, cv2.COLOR_YCR_CB2BGR)
    detail_enhanced = cv2.cvtColor(detail_enhanced, cv2.COLOR_YCR_CB2BGR)

    utils.show_images([img, m_show], ['original', 'M'])
    utils.show_images([img, detail_enhanced], ['original', 'detail enhanced'])

    cv2.imwrite('m_' + filename, m_show)
    cv2.imwrite('de_' + filename, detail_enhanced)

