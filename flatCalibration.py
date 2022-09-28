import numpy as np
import cv2 as cv
import glob
import logging
import copy

# FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS

chessboardSize = (7, 4)
frameSize = (640, 480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 22
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Images/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # print(corners)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

img = cv.imread('Images/9.png')  # This image defines the World Coordinates
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort image
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

# Reprojection Error
# mean_error = 0

# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
#
# print( "total error: {}".format(mean_error/len(objpoints)))

imgTest = cv.imread('caliResult1.png')
gray = cv.cvtColor(imgTest, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
#print(corners)

# If found, add object points, image points (after refining them)
if ret == True:

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)

    # Draw and display the corners
    cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

list1 = []
for elements in objp:
    elements = elements[:2]
    list1.append(elements)

mylist = list(zip(list1, np.squeeze(corners2)))
features_mm_to_pixels_dict = {tuple(mylist[0][0]) : tuple(mylist[0][1]),
                              tuple(mylist[6][0]) : tuple(mylist[6][1]),
                              tuple(mylist[27][0]) : tuple(mylist[27][1]),
                              tuple(mylist[3][0]) : tuple(mylist[3][1])}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def draw_crosshair(image, center, width, color):
    cv.line(image, (center[0] - width//2, center[1]), (center[0] + width//2, center[1]), color, 2)
    cv.line(image, (center[0], center[1] - width//2), (center[0], center[1] + width//2), color, 2)

image_filepath = "caliResult1.png"

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

test_XY_2 = (66, 66, 1)  # Nhap toa do test
test_xy_2 = mm_to_pixels_transformation_mtx @ test_XY_2
image = cv.imread(image_filepath)
annotated_img = copy.deepcopy(image)
draw_crosshair(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), 60, (0, 0, 255))
cv.imwrite("./outputs/annotatedFull.png", annotated_img)