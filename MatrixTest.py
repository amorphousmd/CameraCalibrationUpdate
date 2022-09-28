import numpy as np
import math

origMatrix = np.array([[1],
                       [0],
                       [1]])
theta = 0
rotationMatrix = np.array([[math.cos(math.radians(theta)), -math.sin(math.radians(theta)), 0],
                        [math.sin(math.radians(theta)), math.cos(math.radians(theta)), 0],
                        [0, 0, 1]])
print(rotationMatrix @ origMatrix)
