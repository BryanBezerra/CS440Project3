# Generate a 20x20 Grid:

# Encode the Grid:
#     Use one-hot encoding to represent the colors. Assign a unique one-hot vector to each color.

# Loss Function
#   Use the binary cross entropy
#   -y_i * ln(f(x_i)) - (1-y_i) * ln(1 - f(x_i))

# Output space
#  y -> 0 or 1 (safe or dangerous)

# Model Space
#   Linear regression
#   f(x) = sigmoid( w.x ) w is a vector of size 1600

# Training algorithm
#   SGD
#   x = (1, x_1, x_2, x_3)
#   general linear function
#   x_1 * w_1 + x_2 * w_2 + x_3 * w_3


import numpy as np
from BombDiagram import BombDiagram
from MLFunctions import *

# Writeup needs performance of the model at 2000, 2500, 3000, and 5000 samples.
# Samples get generated pretty much instantly, don't have to worry about saving them, just make new ones each time.
samples = []
NUM_SAMPLES = 5000
DIAGRAM_SIZE = 20
for i in range(NUM_SAMPLES):
    samples.append(BombDiagram(DIAGRAM_SIZE))


# Use the get_flat_image() BombDiagram method to get a flat array of the diagram with a 1 appended to the front
# TODO: Use the generated samples to train the model


# # just setting update to print out everything
# np.set_printoptions(threshold=np.inf)
# print(allGrids[0])
# print(encode_grid(allGrids[0], color_to_one_hot))
#
#
# ## MODEL SPACE
#
# numFeatures = len(allGrids)

# weightVector = np.random.randn(numFeatures, 1)
#
#










# linear regression model with forward passing
# x = input space
# y = output space
# learningRate = size of the steps 
# 

