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

