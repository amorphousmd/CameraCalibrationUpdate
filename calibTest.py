from getTransformationMatrix import *
import math

# Calculate the Transformation Matrix
mm_to_pixels_transformation_mtx = getTransformationMatrix('Images/0.png')
# Enter Translation Value Here
centerPointW = (0, 0, 1)
centerPointP = mm_to_pixels_transformation_mtx @ centerPointW
translationVector = [[-centerPointP[0]],
                     [-centerPointP[1]],
                     [0]]
translationVector = [[0],
                     [0],
                     [0]]
left = np.array([[0, 0],
                 [0, 0],
                 [0, 0]])
# Enter Rotation Value Here
theta = 0
rotationMatrix = np.array([[math.cos(math.radians(theta)), -math.sin(math.radians(theta)), 0],
                        [math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0],
                        [0, 0, 1]])
translationMatrix = np.c_[left, translationVector]
mm_to_pixels_transformation_mtx += translationMatrix
mm_to_pixels_transformation_mtx = mm_to_pixels_transformation_mtx @ rotationMatrix
print(mm_to_pixels_transformation_mtx)


# Graph drawing
image = cv.imread('caliResult1.png')
annotated_img = copy.deepcopy(image)
centerPointW = (0, 0, 1)
centerPointP = mm_to_pixels_transformation_mtx @ centerPointW
draw_crosshair(annotated_img, (round(centerPointP[0]), round(centerPointP[1])), 40, (0, 0, 255))

test_XY_2 = (0, 22, 1)
for i in range(1, 10):
    t2 = tuple(ti * i for ti in test_XY_2[0:2])
    t2 = (*t2, 1)
    test_xy_2 = mm_to_pixels_transformation_mtx @ t2
    cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=3, color=(0, 255, 255), thickness=-1)

test_XY_2 = (22, 0, 1)
for i in range(1, 10):
    t2 = tuple(ti * i for ti in test_XY_2[0:2])
    t2 = (*t2, 1)
    test_xy_2 = mm_to_pixels_transformation_mtx @ t2
    cv.circle(annotated_img, (round(test_xy_2[0]), round(test_xy_2[1])), radius=3, color=(255, 0, 0), thickness=-1)

cv.imwrite("./outputs/annotatedFull.png", annotated_img)
