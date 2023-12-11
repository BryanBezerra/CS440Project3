from CS440Project3.BombDiagram import BombDiagram
import numpy as np

class firstTask:
    # it turns the result of dot product into a probability thats between 0 to 1.
    # activation function (non-linear function)
    
    def __init__(self):
        self.numFeatures = len(BombDiagram(20).get_flat_image)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    # f(x_1,x_2,...,x_d) = sigmoid(w_0 + x_1 w_1 + x_2 w_2 + ... + w_d x_d)
    def predict(self, inputVector, weightVector):
        result = np.dot(inputVector, weightVector)
        return self.sigmoid(result)


    # loss = -y_i * log(f(x_i)) - (1-y_i) * log(1 - f(x_i))
    def binary_cross_entropy_loss(self, yTrue, yPredicted):
        # small constant to avoid log(0)
        epsilon = 1e-15
        loss = -yTrue * np.log(
            np.clip(yPredicted, epsilon, 1 - epsilon) - (1 - yTrue) * np.log(np.clip(1 - yPredicted, epsilon, 1 - epsilon)))
        return loss


    # not too sure about this but it should represent the gradient as below
    # −y_i ​log(f(x_i​))−(1−y_i​)log(1−f(x_i​))
    def gradient_binary_cross_entropy(self, yTrue, yPredicted, x_j):
        gradient = -yTrue * (1 - yPredicted) * x_j + (1 - yTrue) * yPredicted * x_j
        return gradient


    def linear_regression_loss(self, y_dang_or_not, f_x, weight):
        epsilon = 1e-15
        # loss_fun = np.sum(((-1) * y_dang_or_not * np.log(model_space(f_x, weight)+epsilon) - (1 - y_dang_or_not) * (1 - model_space(f_x, weight)+epsilon)))
        loss_fun = np.mean(np.sum(np.power((np.dot(f_x, weight) - y_dang_or_not), 2)))  # linear regression loss
        return loss_fun
    
    # TODO make weight range from [-0.025, 0.025]
    def weight_initialization(self, numFeatures):
        return np.random.randn(numFeatures, 1)

    def train_linear_regression_sgd(self, x, y, learningRate, numEpochs, starting_weights):
        numExamples = x.shape

        for epoch in range(numEpochs):

            for i in range(numExamples):
                # random data point
                random_index = np.random.randint(numExamples)
                observed_in = x[random_index:random_index + 1, :]
                observed_out = y[random_index:random_index + 1]

                # forward pass
                prediction = self.sigmoid(self.predict(observed_in, weights))

                loss = self.binary_cross_entropy_loss(observed_out, prediction)

                gradient = self.gradient_binary_cross_entropy(observed_out, prediction, observed_in.T)

                # update weights using stochastic gradient descent
                weights -= learningRate * gradient

        return weights, loss

    def stochastic_gradient_descent(self, samples, diagram_size, num_epochs, alpha, bomb_diagram):
        (num_image,) = np.shape(samples)
        # index = np.random.randint(num_image)

        #TODO make weight range from [-0.025, 0.025]
        weight = np.random.randn(diagram_size * diagram_size * 4 + 1, 1) * 0.001
        print(weight)

        for epoch in range(num_epochs):
            for i in range(num_image):
                random_x = np.random.randint(num_image)
                input = samples[random_x]
                observed_in = [BombDiagram.get_flat_image(input)]
                observed_out = BombDiagram.is_dangerous(input)
                if observed_out:
                    observed_out = 1
                else:
                    observed_out = 0
                # fun_x = model_space(input_x, weight)
                z = (np.dot(observed_in, weight))
                print("loss: ", self.linear_regression_loss(observed_out, observed_in, weight), end="\n")

                weight = self.sigmoid(z)  # SGD with linear regression
                # weight = weight -

                # print("\n", weight)

        return weight
