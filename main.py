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


import random
import numpy as np

# in the last lecture he was saying that # of samples should be greater or equal to the # of features so we can start from 1601 and increase it later
numSamples = 1601

# one hot vectors could also be represented in a way such that U = (0,0,0,0)

color_to_one_hot = {
    "R": (0, 0, 0, 1),
    "B": (0, 0, 1, 0),
    "Y": (0, 1, 0, 0),
    "G": (1, 0, 0, 0),
    "U": (0, 0, 0, 0)
}


def generate_grid(grid_size):
    grid = np.full((grid_size, grid_size), "U", dtype=str)
    return grid


def getRandomHotColor(random_color):
    # getting the random hot vector color
    random_hot_vector = color_to_one_hot[random_color]
    return random_hot_vector


def RowColoring(grid, colorsForRow):
    row_index = random.choice(sizeForRow)
    random_color = random.choice(colorsForRow)
    # cursorOne = random_color
    # if cursorOne = "R":
    # check if cursorTwo = "Y"
    colorsForRow.remove(random_color)
    sizeForRow.remove(row_index)
    grid[row_index, :] = random_color
    return grid


def ColumnColoring(grid, colorsForCol):
    col_index = random.choice(sizeForCol)
    random_color = random.choice(colorsForCol)
    colorsForCol.remove(random_color)
    sizeForCol.remove(col_index)
    grid[:, col_index] = random_color
    return grid


# to-do: we need to add 1 to beginning for bias term
# to-do2: we need to create a class that takes two inputs, the data and the label (safe or dangerous)
# arrays of size 1600 vectors
def encode_grid(grid, color_to_one_hot):
    # Flatten the grid and encode each color using one-hot vectors
    encoded_vector = []

    for row in grid:
        for cell in row:
            encoded_vector.extend(color_to_one_hot[cell])

    return np.array(encoded_vector)


allGrids = []
## when we create the data does every data need to be unique?

for i in range(numSamples):
    colorsFor = ["R", "B", "Y", "G"]
    sizeForRow = list(range(0, 20))
    sizeForCol = list(range(0, 20))

    # generating the grid
    grid = generate_grid(20)
    # filling the first row
    grid = RowColoring(grid, colorsFor)
    # filling first col
    grid = ColumnColoring(grid, colorsFor)
    # filling second row
    grid = RowColoring(grid, colorsFor)
    # filling second col
    grid = ColumnColoring(grid, colorsFor)

    allGrids.append(grid)

# just setting update to print out everything
np.set_printoptions(threshold=np.inf)
print(allGrids[0])
print(encode_grid(allGrids[0], color_to_one_hot))

## needs the labeling the data whether its safe or dangerous


# Function to label the grid as safe or dangerous
# def label_grid(grid):
# if is_dangerous(grid):
#        return "Dangerous"
#    else:
#        return "Safe"


## MODEL SPACE

numFeatures = len(allGrids)

weightVector = np.random.randn(numFeatures, 1)


# f(x_1,x_2,...,x_d) = sigmoid(w_0 + x_1 w_1 + x_2 w_2 + ... + w_d x_d)
def predict(inputVector, weightVector):
    result = np.dot(inputVector, weightVector)
    return sigmoid(result)


# it turns the result of dot product into a probability thats between 0 to 1.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# loss = -y_i * log(f(x_i)) - (1-y_i) * log(1 - f(x_i))
def binary_cross_entropy_loss(yTrue, yPredicted):
    # small constant to avoid log(0)
    epsilon = 1e-15
    loss = -yTrue * np.log(
        np.clip(yPredicted, epsilon, 1 - epsilon) - (1 - yTrue) * np.log(np.clip(1 - yPredicted, epsilon, 1 - epsilon)))
    return loss


# not too sure about this but it should represent the gradient as below
# −y_i ​log(f(x_i​))−(1−y_i​)log(1−f(x_i​))
def gradient_binary_cross_entropy(yTrue, yPredicted, x_j):
    gradient = -yTrue * (1 - yPredicted) * x_j + (1 - yTrue) * yPredicted * x_j
    return gradient
