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



# f(x_1,x_2,...,x_d) = sigmoid(w_0 + x_1 w_1 + x_2 w_2 + ... + w_d x_d)
def predict(inputVector, weightVector):
     result = np.dot(inputVector, weightVector)
     return sigmoid(result)


# it turns the result of dot product into a probability thats between 0 to 1.
# activation function (non-linear function)
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

# linear regression model with forward passing
# x = input space
# y = output space
# learningRate = size of the steps 
# 
def train_linear_regression_sgd(x, y, learningRate, numEpochs):
    numExamples, numFeatures = x.shape
    weights = np.random.randn(numFeatures, 1)
    for epoch in range(numEpochs):
        

        for i in range(numExamples):
            # random data point
            random_index = np.random.randint(numExamples)
            x_i = x[random_index:random_index + 1, :]
            y_i = y[random_index:random_index + 1]

            # forward pass
            prediction = sigmoid(predict(x_i, weights))

            loss = binary_cross_entropy_loss(y_i, prediction)

            gradient = gradient_binary_cross_entropy(y_i, prediction, x_i.T)

            # update weights using stochastic gradient descent
            weights -= learningRate * gradient

        
    return weights