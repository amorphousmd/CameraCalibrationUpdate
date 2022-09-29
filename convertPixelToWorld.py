# convertPixelToWorld.py
# Returns the world coordinates from pixel
# Input: Tuple. Example (x, y)
# Output: Tuple. Example (X, Y)
import numpy as np

pixelToWorldMatrix = np.load('calibSaves/PtWMatrix.npy')  # Load transformation matrix from file


def convertPixelToWorld(planeMatrix):
    planeMatrix = (*planeMatrix, 1)  # Append a 1 to tuple
    # print(pixelToWorldMatrix)  # Print T matrix, can uncomment this
    try:
        worldMatrix = pixelToWorldMatrix @ planeMatrix  # Calculate world matrix
    except ValueError:
        print("Wrong input size. Make sure the input form is (x,y)")
    else:
        output = (worldMatrix[0], worldMatrix[1])  # Remove the 1 in matrix
        return output
