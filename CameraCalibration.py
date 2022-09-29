import numpy as np
import cv2 as cv
import glob
import copy
import sys
import math


def draw_crosshair(image, center, width, color):
    cv.line(image, (center[0] - width // 2, center[1]), (center[0] + width // 2, center[1]), color, 3)
    cv.line(image, (center[0], center[1] - width // 2), (center[0], center[1] + width // 2), color, 3)

# Used to convert 1x3 rotation matrix to 3x3 rotation matrix
def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat


# Chess board definition
chessboardSize = (7, 4)
frameSize = (640, 480)

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 22
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Get calibration images

images = glob.glob('./Images/*.png')

# Get imgpoints (the corners of the chessboard)
for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

cv.destroyAllWindows()

# Get all the required matrix here
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Get the transformation matrix
rotationMatrix = rodrigues_vec_to_rotation_mat(np.squeeze(rvecs[0]))
extractedData = rotationMatrix[:, [0, 1]]
deprecatedMatrix = np.c_[np.squeeze(extractedData), np.squeeze(tvecs[0])]
homographyMatrix = np.squeeze(cameraMatrix) @ deprecatedMatrix
transformationMatrix = homographyMatrix/(np.squeeze(tvecs[0][2]))
print('\n World to Pixel Transformation Matrix: \n', transformationMatrix)
print('\n Pixel to World Transformation Matrix: \n', np.linalg.inv(transformationMatrix))
np.save('./calibSaves/WtPMatrix.npy', transformationMatrix)
np.save('./calibSaves/PtWMatrix.npy', np.linalg.inv(transformationMatrix))
print('Transformation matrices written at ./calibSaves')

# Test the result on a file (See ./outputs/annotatedFullUpdate.png)
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
