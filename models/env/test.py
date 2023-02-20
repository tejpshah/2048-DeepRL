import numpy as np

# create a 5x5 matrix filled with zeros
matrix = np.zeros((5, 5))

# find all locations in the matrix where the value is zero
zero_locations = np.argwhere(matrix == 0)

print(zero_locations)

print(np.log2([4,8,16]).sum())