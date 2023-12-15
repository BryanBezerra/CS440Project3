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
        non_linear_features_near_column_product = self.diagram_size / 2
        num_features = self.diagram_size * self.diagram_size * self.NUM_COLORS + 1 + (int)(non_linear_features_near_column_product)
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

    def add_non_linear_feature(self, bomb_diagram):
        """Adding non-linear feature and append the variables at the end of the flatten image vector

        Since this part is asking for which wire to cut it is useful to consider the column of the image
        and find the relationship between column which is off vertical wires.

        General idea:
            Image one hot vector:
            Linear part:                         Non-Linear part:
            (col 1) (col 2) (col 3) ... (col N) (col 1 * col 2) (col 3 * col 4) ... (col i * col j)
            x_1     x_2     x_3         x_N     (x_1 * x_2)     (x_3 * x_4)         (x_i * x_j)

        Args:
            bomb_diagram: the domb diagram used to calculate the loss
        """
        
        image_array = bomb_diagram.get_image()
        image_x = bomb_diagram.get_flat_image()
        (_,col,_) = np.shape(image_array)
        temp = []
        
        # add dot_product(x_i, x_j) for very i + 1 = j as column index
        for i in range(0, col, 2):
            temp_dot_prod = np.sum(image_array[:,i] * image_array[:,i+1])
            temp.append(temp_dot_prod)
            
        return np.concatenate((image_x, temp))

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
        epsilon = 1e-15
        predicted_out = self.multiclass_classification(bomb_diagram)
        observed_out = np.array(bomb_diagram.get_wire_to_cut().value)
        # Categorical Cross Entropy Loss
        return np.sum(-np.log(np.dot(predicted_out, observed_out) + epsilon)) / len(observed_out)

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

    def multiclass_classification(self, bomb_diagram):
        """Applies softmax classification to predict which wire should be cut based on the bomb diagram.

        Args:
            bomb_diagram: the diagram to be analyzed

        Returns:
            A numpy array of four probabilities that each wire should be cut. The sum of the probabilities is 1.
            [P(green), P(yellow), P(blue), P(red)]
        """
        input_vector = self.add_non_linear_feature(bomb_diagram)
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
        """Uses the model to predict which wire needs to be cut to disarm the bomb.

        Args:
            bomb_diagram: the bomb diagram to be predicted

        Return:
            the color of the wire to cut
        """
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

    def train_on_one_sample(self, alpha, regularization_lambda):
        """Trains the model on a single datapoint.

        Args:
            alpha: the learning rate of the model, must be > 0
            regularization_lambda: higher number creates preference for smaller weights
        """
        data_point = self.samples.pop()
        self.used_samples.append(data_point)
        prediction = self.multiclass_classification(data_point)
        flat_image = self.add_non_linear_feature(data_point)
        # Green Weights
        loss_gradient = (prediction[0] - data_point.get_wire_to_cut().value[0]) * flat_image
        ridge_regularization = 2 * regularization_lambda * np.sum(self.weights_green)
        self.weights_green = self.weights_green - alpha * (loss_gradient * ridge_regularization)
        # Yellow Weights
        loss_gradient = (prediction[1] - data_point.get_wire_to_cut().value[1]) * flat_image
        ridge_regularization = 2 * regularization_lambda * np.sum(self.weights_yellow)
        self.weights_yellow = self.weights_yellow - alpha * (loss_gradient * ridge_regularization)
        # Blue Weights
        loss_gradient = (prediction[2] - data_point.get_wire_to_cut().value[2]) * flat_image
        ridge_regularization = 2 * regularization_lambda * np.sum(self.weights_blue)
        self.weights_blue = self.weights_blue - alpha * (loss_gradient * ridge_regularization)
        # Red Weights
        loss_gradient = (prediction[3] - data_point.get_wire_to_cut().value[3]) * flat_image
        ridge_regularization = 2 * regularization_lambda * np.sum(self.weights_red)
        self.weights_red = self.weights_red - alpha * (loss_gradient * ridge_regularization)

    def train_model_stochastic(self, num_steps, alpha, reg_lambda, loss_calc_freq, show_training_data=True):
        """Uses stochastic gradient to train the model on the training data.

        Args:
            num_steps: how many times the model will be trained on a random data point
            alpha: the learning rate of the model, must be > 0
            reg_lambda: higher number creates preference for smaller weights
            loss_calc_freq: how often the program calculates and prints the current loss on all sample and test data
            show_training_data: true if the program is to print and graph loss data, otherwise false; defaults true
        """
        training_loss = []
        testing_loss = []
        for step in range(num_steps):
            if len(self.samples) == 0:
                self.refresh_samples()

            self.train_on_one_sample(alpha, reg_lambda)

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
