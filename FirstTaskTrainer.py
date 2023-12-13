import math

import numpy as np
import matplotlib.pyplot as plt
import random
from BombDiagram import BombDiagram


class FirstTaskTrainer:
    """Trains a logistic regression model using stochastic gradient descent and predicts whether a given bomb diagram
    is dangerous.

    Attributes:
         diagram_size: the size of the bomb diagrams accepted by the model
         weights: the weights of the model
         samples: the samples used to train the model
         used_samples: samples the model has already trained on during the current cycle
         independent_test_data: independent data used only to test the model, never to train it
    """
    NUM_COLORS = 4  # number of different wire colors

    def __init__(self, diagram_size=20):
        """Initializes the instance based on diagram size and sets starting weights.

        Args:
            diagram_size: the size of the bomb diagrams accepted by the model
        """
        self.diagram_size = diagram_size
        self.weights = self.generate_starting_weights()
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
        start_magnitude = (num_features ** 0.5) ** -1
        return np.random.uniform(-start_magnitude, start_magnitude, num_features)

    def add_generated_samples(self, num_samples):
        """Generates samples to train the model and a separate set of samples to test the model.

        Args:
            num_samples: the number of samples added to the training and testing data
        """
        for i in range(num_samples):
            self.samples.append(BombDiagram(self.diagram_size))
            self.independent_test_data.append(BombDiagram(self.diagram_size))

    def clear_samples(self):
        """Clears the training and testing data"""
        self.samples.clear()
        self.independent_test_data.clear()

    def sigmoid(self, bomb_diagram):
        """Applies the sigmoid function to the dot product of the model weights and the bomb diagram

        Args:
            bomb_diagram: a bomb diagram of the appropriate size

        Returns:
            the result of the sigmoid function applied to the dot product of the model weights and bomb diagram
        """
        input_vec = bomb_diagram.get_flat_image()
        return 1 / (1 + math.e ** (-np.dot(input_vec, self.weights)))

    def predict(self, bomb_diagram):
        """Uses the model to predict whether the bomb diagram is dangerous or not.

        Args:
            bomb_diagram: the bomb diagram to be predicted

        Return:
            1 if the bomb is dangerous, otherwise 0
        """
        predicted_value = self.sigmoid(bomb_diagram)
        if predicted_value < 0.5:
            return 0
        else:
            return 1

    def loss(self, bomb_diagram):
        """Calculates the loss of the model on a single sample using the log loss function.

        Args:
            bomb_diagram: the domb diagram used to calculate the loss

        Returns:
            the loss of the function based on the bomb diagram and the predicted result
        """
        predicted_val = self.sigmoid(bomb_diagram)
        observed_val = bomb_diagram.is_dangerous()
        return -observed_val * math.log(predicted_val) - (1 - observed_val) * math.log(1 - predicted_val)

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

    def refresh_samples(self):
        """Returns the used samples to unused samples and shuffles the sample order."""
        self.samples = self.samples.copy() + self.used_samples.copy()
        random.shuffle(self.samples)
        self.used_samples.clear()

    def train_on_one_sample(self, alpha):
        """Trains the model on a single datapoint.

        Args:
            alpha: the learning rate of the model, must be > 0
        """
        data_point = self.samples.pop()
        self.used_samples.append(data_point)
        prediction = self.predict(data_point)
        self.weights = self.weights - alpha * (prediction - data_point.is_dangerous()) * data_point.get_flat_image()

    def train_model_stochastic(self, num_steps, alpha, loss_calc_freq, show_training_data=True):
        """Uses stochastic gradient to train the model on the training data.

        Args:
            num_steps: how many times the model will be trained on a random data point
            alpha: the learning rate of the model, must be > 0
            loss_calc_freq: how often the program calculates and prints the current loss on all sample and test data
            show_training_data: true if the program is to print and graph loss data, otherwise false; defaults true
        """
        training_loss = []
        testing_loss = []
        for i in range(num_steps):
            if len(self.samples) == 0:
                self.refresh_samples()

            self.train_on_one_sample(alpha)

            if show_training_data:
                if i % loss_calc_freq == 0:
                    training_loss.append(self.calc_loss_on_samples())
                    testing_loss.append(self.calc_loss_on_independent_data())
                    print("Step", i, "Loss:", training_loss[-1], "Loss on independent data:", testing_loss[-1])

        if show_training_data:
            training_loss.append(self.calc_loss_on_samples())
            testing_loss.append(self.calc_loss_on_independent_data())
            print("Step", num_steps, "Loss:", training_loss[-1], "Loss on independent data:", testing_loss[-1])
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
