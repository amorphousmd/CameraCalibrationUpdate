import numpy as np

# Returns the world coordinates
# Input: Tuple. Example (x, y)
# Output: Tuple. Example (X, Y)


def convertPixelToWorld(planeMatrix):
    planeMatrix = (*planeMatrix, 1)  # Append a 1 to tuple
    pixelToWorldMatrix = np.load('calibSaves/TMatrix.npy')  # Load transformation matrix from file
    # print(pixelToWorldMatrix)  # Print T matrix, can uncomment this
    try:
        worldMatrix = pixelToWorldMatrix @ planeMatrix  # Calculate world matrix
    except ValueError:
        print("Wrong input size. Make sure the input form is (x,y)")
    else:
        output = (worldMatrix[0], worldMatrix[1])  # Remove the 1 in matrix
        return output

# Example Testing Section


planeMatrixTest1 = (300, 400)
print(convertPixelToWorld(planeMatrixTest1))

# planeMatrixTest1 = (303, 220, 1)
# print(convertPixelToWorld(planeMatrixTest1))
