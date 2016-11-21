import cv2
import math
import numpy as np


def drawAxis(img, start_pt, vec, colour, length):
    CV_AA = 16

    end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

    cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)

    cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);

    angle = math.atan2(vec[1], vec[0])
    print(angle)

    qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4));
    qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4));
    cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA);

    qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4));
    qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4));
    cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA);


if __name__ == '__main__':
    src = cv2.imread("images.jpeg")

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img, contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):

        area = cv2.contourArea(contours[i])

        if area < 1e2 or 1e5 < area:
            continue

        cv2.drawContours(src, contours, i, (0, 0, 255), 2, 8, hierarchy, 0)

        X = np.array(contours[i], dtype=np.float).reshape((contours[i].shape[0], contours[i].shape[2]))

        mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float), maxComponents=1)

        pt = (mean[0][0], mean[0][1])
        vec = (eigenvectors[0][0], eigenvectors[0][1])
        drawAxis(src, pt, vec, (255, 255, 0), 150)


    cv2.imshow('output', src)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
