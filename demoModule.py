import cv2 as cv
from convertPixelToWorld import *
from decimal import Decimal

# planeMatrixTest = (300, 400)
# print(convertPixelToWorld(planeMatrixTest))


prevX, prevY = -1, -1


def printCoordinate(event, x, y, flags, params):
    global prevX, prevY
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), 3, (0, 0, 255), -1)

        font = cv.FONT_HERSHEY_PLAIN
        planeMatrix = (x, y)
        worldMatrix = convertPixelToWorld(planeMatrix)
        print('Pixel Location: ' +'(' + str(x) + ',' + str(y) + ')')
        strXY = '(' + str(round(Decimal(str(worldMatrix[0])), 2)) + ',' + str(
            round(Decimal(str(worldMatrix[1])), 2)) + ')'
        cv.putText(img, strXY, (x + 10, y - 10), font, 1, (255, 0, 255), 2)
        if prevX == -1 and prevY == -1:
            prevX, prevY = x, y
        else:
            prevX, prevY = -1, -1
        cv.imshow("image", img)


img = cv.imread('./Images/0.png')

cv.imshow("image", img)
cv.setMouseCallback("image", printCoordinate)
cv.waitKey()
cv.destroyAllWindows()