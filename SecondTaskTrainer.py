# Input Space 
# Same as the first task which is vector size of 1601

# Output Space
# The color of the third laid down wire. Blue, Yellow, Green or Red 
# P(of being Blue) - P(of being Red) - P(of being Yellow) - P(of being Green)
# Probability of belonging to each class

# Model Space
# Multi Class Classification
# W_a = weight vector associated with a - W_b = weight vector associated with b  - W_c = weight vector associated with c
# F(x) = ( e^(W_a . x)/SUM_k(e^(w_k . x))  ,   e^(W_b . x)/SUM_k(e^(w_k . x))  ,  e^(W_c . x)/SUM_k(e^(w_k . x)) )
# F(x) is a three dimensional output function of our input

# Loss Function
# Categorical cross entropy loss (softmax function)

# Training Algorithm
# Grimport sys

import numpy as np
import matplotlib.pyplot as plt
import random
from BombDiagram import BombDiagram
from WireColor import WireColor


class SecondTaskTrainer:
    NUM_COLORS = 4

    def __init__(self, diagram_size=20):
        """Initializes the instance based on diagram size and sets starting weights.

            Args:
                diagram_size: the size of the bomb diagrams accepted by the model
            """
        self.diagram_size = diagram_size
        self.weights_red = self.generate_starting_weights()
        self.weights_blue = self.generate_starting_weights()
        self.weights_green = self.generate_starting_weights()
        self.weights_yellow = self.generate_starting_weights()

        self.samples = []
        self.used_samples = []
        self.independent_test_data = []

    def generate_starting_weights(self):
        """Generates the model's starting weights.

        Generation is based on the size of the diagram in an attempt to keep weights relatively low.

        Returns:
            A numpy array containing the starting weights for the model.
        """
        num_features = self.diagram_size * self.diagram_size * self.NUM_COLORS + 1
        start_magnitude = ((num_features ** 0.5) ** -1)
        return np.random.uniform(-start_magnitude, start_magnitude, num_features)

    def add_generated_samples(self, num_samples):
        """Generates samples to train the model and a separate set of samples to test the model.

        Args:
            num_samples: the number of samples added to the training and testing data
        """
        for i in range(num_samples):
            self.samples.append(BombDiagram(self.diagram_size, True))
            self.independent_test_data.append(BombDiagram(self.diagram_size, True))

    def refresh_samples(self):
        self.samples = self.samples.copy() + self.used_samples.copy()
        random.shuffle(self.samples)
        self.used_samples.clear()

    def loss(self, bomb_diagram):
        """Calculates the loss of the model on a single sample using the log loss function.

        Args:
            bomb_diagram: the domb diagram used to calculate the loss

        Returns:
            the loss of the function based on the bomb diagram and the predicted result
        """
        predicted_out = self.multiclass_classification(bomb_diagram)
        observed_out = np.array(bomb_diagram.get_wire_to_cut().value)
        # Categorical Cross Entropy Loss
        return np.sum(observed_out * (-np.log(predicted_out)))

    def calc_loss_on_samples(self):
        """Calculates the mean loss of the model based on the training data.

        Returns:
            the mean loss of the model based on the training data
        """
        loss_sum = 0
        count = 0
        for sample in self.samples:
            loss_sum += self.loss(sample)
            count += 1
        for sample in self.used_samples:
            loss_sum += self.loss(sample)
            count += 1
        return loss_sum / count

    def calc_loss_on_independent_data(self):
        """Calculates the mean loss of the model based on the independent testing data.

        Returns:
            the mean loss of the model based on the independent testing data
        """
        loss_sum = 0
        for sample in self.independent_test_data:
            loss_sum += self.loss(sample)
        return loss_sum / len(self.independent_test_data)

    # probabilities : [0.2395125813000004, 0.2290498266336638, 0.3053224292131634, 0.2261151628531726]
    def multiclass_classification(self, bomb_diagram):
        input_vector = bomb_diagram.get_flat_image()
        dot_product_red = np.dot(input_vector, self.weights_red)
        dot_product_blue = np.dot(input_vector, self.weights_blue)
        dot_product_green = np.dot(input_vector, self.weights_green)
        dot_product_yellow = np.dot(input_vector, self.weights_yellow)

        exp_red = np.exp(dot_product_red)
        exp_blue = np.exp(dot_product_blue)
        exp_green = np.exp(dot_product_green)
        exp_yellow = np.exp(dot_product_yellow)

        denominator = exp_red + exp_blue + exp_green + exp_yellow

        softmax_red = exp_red / denominator
        softmax_blue = exp_blue / denominator
        softmax_green = exp_green / denominator
        softmax_yellow = exp_yellow / denominator

        # Return the four-dimensional output which are the softmax probabilities of each class
        output = np.array([softmax_green, softmax_yellow, softmax_blue, softmax_red])
        return output

    def predict(self, bomb_diagram):
        predicted_values = self.multiclass_classification(bomb_diagram)
        result = WireColor.GREEN
        max_probability = predicted_values[0]
        if predicted_values[1] > max_probability:
            max_probability = predicted_values[1]
            result = WireColor.YELLOW
        if predicted_values[2] > max_probability:
            max_probability = predicted_values[2]
            result = WireColor.BLUE
        if predicted_values[3] > max_probability:
            result = WireColor.RED
        return result

    def train_on_one_sample(self, alpha):
        data_point = self.samples.pop()
        self.used_samples.append(data_point)
        prediction = self.multiclass_classification(data_point)
        flat_image = data_point.get_flat_image()
        # Green Weights
        loss_gradient = (prediction[0] - data_point.get_wire_to_cut().value[0]) * flat_image
        self.weights_green = self.weights_green - alpha * loss_gradient
        # Yellow Weights
        loss_gradient = (prediction[1] - data_point.get_wire_to_cut().value[1]) * flat_image
        self.weights_yellow = self.weights_yellow - alpha * loss_gradient
        # Blue Weights
        loss_gradient = (prediction[2] - data_point.get_wire_to_cut().value[2]) * flat_image
        self.weights_blue = self.weights_blue - alpha * loss_gradient
        # Red Weights
        loss_gradient = (prediction[3] - data_point.get_wire_to_cut().value[3]) * flat_image
        self.weights_red = self.weights_red - alpha * loss_gradient

    def stochastic_gradient_descent(self, num_steps, alpha, loss_calc_freq, show_training_data=True):
        training_loss = []
        testing_loss = []
        for step in range(num_steps):
            if len(self.samples) == 0:
                self.refresh_samples()

            self.train_on_one_sample(alpha)

            if show_training_data:
                if step % loss_calc_freq == 0:
                    training_loss.append(self.calc_loss_on_samples())
                    testing_loss.append(self.calc_loss_on_independent_data())
                    print("Step", step, "Loss:", training_loss[-1], "Loss on independent data:", testing_loss[-1])

        if show_training_data:
            training_loss.append(self.calc_loss_on_samples())
            testing_loss.append(self.calc_loss_on_independent_data())
            print("Step", num_steps, "Loss:", training_loss[-1], "Loss on independent data:", testing_loss[-1])
            print("Min training:", np.min(training_loss))
            print("Min test:", np.min(testing_loss))
            self.graph_loss(training_loss, testing_loss, loss_calc_freq, alpha)

    def graph_loss(self, training_loss, testing_loss, loss_calc_freq, alpha, num_x_axis_ticks=10):
        """Graphs the model's loss over time.

        Args:
            training_loss: a list of the model's loss values over time based on the training data
            testing_loss: a list of the model's loss values over time based on the testing data
            loss_calc_freq: how often the data was recorded in steps
            alpha: the learning rate of the model
            num_x_axis_ticks: how many ticks the x-axis label should have; defaults to 10
        """
        data_len = len(training_loss)
        num_samples = len(self.samples) + len(self.used_samples)
        plt.plot(training_loss, label='Learning Data')
        plt.plot(testing_loss, label='Testing Data')
        x_axis_ticks = np.arange(0, data_len, data_len // num_x_axis_ticks)
        x_axis_labels = [str(i * loss_calc_freq) for i in range(0, data_len, (data_len // num_x_axis_ticks))]
        plt.xticks(x_axis_ticks, x_axis_labels)
        plt.xlabel('Number of Steps')
        plt.ylabel('Loss')
        plt.suptitle('Model Loss Over Time')
        title = "at α = " + str(alpha) + " sample size = " + str(num_samples)
        plt.title(title)

        plt.legend()
        plt.show()
