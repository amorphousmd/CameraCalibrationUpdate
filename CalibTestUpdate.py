import numpy as np
import cv2 as cv
import glob
from ConvertRodtoRotationMatrix import *
import copy

def draw_crosshair(image, center, width, color):
    cv.line(image, (center[0] - width // 2, center[1]), (center[0] + width // 2, center[1]), color, 3)
    cv.line(image, (center[0], center[1] - width // 2), (center[0], center[1] + width // 2), color, 3)

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (7,4)
frameSize = (640,480)



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


images = glob.glob('./Images/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# print(cameraMatrix)
print(np.array(cameraMatrix))
rotationMatrix = rodrigues_vec_to_rotation_mat(np.squeeze(rvecs[0]))
print(rotationMatrix)
extractedData = rotationMatrix[:, [0, 1]]
print(extractedData)
print(tvecs[0])
deprecatedMatrix = np.c_[np.squeeze(extractedData), np.squeeze(tvecs[0])]
homographyMatrix = np.squeeze(cameraMatrix) @ deprecatedMatrix
transformationMatrix = homographyMatrix/(np.squeeze(tvecs[0][2]))
print(transformationMatrix)

image = cv.imread('./Images/0.png')
annotated_img = copy.deepcopy(image)
centerPointW = (0, 0, 1)
centerPointP = transformationMatrix @ centerPointW
draw_crosshair(annotated_img, (round(centerPointP[0]), round(centerPointP[1])), 40, (0, 0, 255))

test_XY_2 = (0, 22, 1)
for i in range(1, 10):
    t2 = tuple(ti * i for ti in test_XY_2[0:2])
    t2 = (*t2, 1)
    test_xy_2 = transformationMatrix @ t2
    cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=3, color=(0, 255, 255), thickness=-1)

test_XY_2 = (22, 0, 1)
for i in range(1, 10):
    t2 = tuple(ti * i for ti in test_XY_2[0:2])
    t2 = (*t2, 1)
    test_xy_2 = transformationMatrix @ t2
    cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=3, color=(255, 0, 0), thickness=-1)

cv.imwrite("./outputs/annotatedFullUpdate.png", annotated_img)