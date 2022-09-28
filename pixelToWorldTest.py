# pixelToWorldTest.py

import logging
import cv2
import numpy as np
from decimal import Decimal


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def draw_crosshair(image, center, width, color):
    cv2.line(image, (center[0] - width//2, center[1]), (center[0] + width//2, center[1]), color, 2)
    cv2.line(image, (center[0], center[1] - width//2), (center[0], center[1] + width//2), color, 2)

image_filepath = "./Images/0.png"
features_mm_to_pixels_dict =  {(0., 0.): (178.97382,  42.35252, ),
                              (132.,   0.): (405.64944,  41.79083),
                              (132.,  66.): (407.20145, 155.39421),
                              (0., 66.): (178.95978, 156.45543)}
A = np.zeros((2 * len(features_mm_to_pixels_dict), 6), dtype=float)
b = np.zeros((2 * len(features_mm_to_pixels_dict), 1), dtype=float)
index = 0

for XY, xy in features_mm_to_pixels_dict.items():
    X = XY[0]
    Y = XY[1]
    x = xy[0]
    y = xy[1]
    A[2 * index, 0] = x
    A[2 * index, 1] = y
    A[2 * index, 2] = 1
    A[2 * index + 1, 3] = x
    A[2 * index + 1, 4] = y
    A[2 * index + 1, 5] = 1
    b[2 * index, 0] = X
    b[2 * index + 1, 0] = Y
    index += 1
# A @ x = b
x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

pixels_to_mm_transformation_mtx = np.array([[x[0, 0], x[1, 0], x[2, 0]], [x[3, 0], x[4, 0], x[5, 0]], [0, 0, 1]])
logging.debug("main(): pixels_to_mm_transformation_mtx = \n{}".format(pixels_to_mm_transformation_mtx))

mm_to_pixels_transformation_mtx = np.linalg.inv(pixels_to_mm_transformation_mtx)

prevX, prevY = -1, -1


def printCoordinate(event, x, y, flags, params):
    global prevX, prevY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        font = cv2.FONT_HERSHEY_PLAIN
        planeMatrix = (x, y, 1)
        worldMatrix = pixels_to_mm_transformation_mtx @ planeMatrix
        print('(' + str(x) + ',' + str(y) + ')')
        strXY = '(' + str(round(Decimal(str(worldMatrix[0])), 2)) + ',' + str(
            round(Decimal(str(worldMatrix[1])), 2)) + ')'
        cv2.putText(img, strXY, (x + 10, y - 10), font, 1, (255, 0, 255), 2)
        if prevX == -1 and prevY == -1:
            prevX, prevY = x, y
        else:
            prevX, prevY = -1, -1
        cv2.imshow("image", img)


img = cv2.imread('Images/0.png')

cv2.imshow("image", img)
cv2.setMouseCallback("image", printCoordinate)
cv2.waitKey()
cv2.destroyAllWindows()




