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
            self.weights = [self.weights_red, self.weights_blue, self.weights_green, self.weights_yellow]
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

    def loss(self, bomb_diagram):
        """Calculates the loss of the model on a single sample using the log loss function.

        Args:
            bomb_diagram: the domb diagram used to calculate the loss

        Returns:
            the loss of the function based on the bomb diagram and the predicted result
        """
        image_x = bomb_diagram.get_flat_image()
        predicted_val = self.multiclass_classification(image_x, self.weights_red, self.weights_blue, self.weights_green, self.weights_yellow)
        observed_val = np.array(BombDiagram.get_wire_to_cut(bomb_diagram).value) # true_label_y = np.array(BombDiagram.get_wire_to_cut(input).value)
        return self.categorical_cross_entropy_loss(predicted_val, observed_val)

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

    # − ∑_c=1 I{y = c} ln p_c
    def categorical_cross_entropy_loss(self,y_true, y_predicted):
        epsilon = 1e-15
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_predicted))
        return loss

    # probabilities : [0.23451, 0.2414556, 0.13134, 0.25516]
    def multiclass_classification(self, input_vector, weights_red, weights_blue, weights_green, weights_yellow):
        dot_product_red = np.dot((input_vector), weights_red)
        dot_product_blue = np.dot(input_vector, weights_blue)
        dot_product_green = np.dot(input_vector, weights_green)
        dot_product_yellow = np.dot(input_vector, weights_yellow)

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
        output = [softmax_red, softmax_blue, softmax_green, softmax_yellow]

        return output

    def stochastic_gradient_desecent(self, alpha ,num_steps):

        training_loss = []
        testing_loss = []
        for step in range(num_steps):
            # random select a datapoint
            random_index = np.random.randint(len(self.samples))
            input = self.samples[random_index]
            # input = self.samples.pop()
            # self.used_samples.append(input)
            image_x = BombDiagram.get_flat_image(input)
            true_label_y = np.array(BombDiagram.get_wire_to_cut(input).value)

            # softmax regression
            output_probs = self.multiclass_classification(image_x, self.weights_red, self.weights_blue, self.weights_green, self.weights_yellow)

            # getting training loss
            # loss = self.categorical_cross_entropy_loss(true_label_y, output_probs)

            # stochastic gradient decsent
            # error = (output_probs - true_label_y)
            # gradient = [[error[0]],[error[1]],[error[2]],[error[3]]] * image_x
            # self.weights = self.weights - alpha * gradient

            # stochastic gradient decsent
            self.weights_red = self.weights_red - alpha * (output_probs[0] - true_label_y[0]) * image_x
            self.weights_blue = self.weights_blue - alpha * (output_probs[1] - true_label_y[1]) * image_x
            self.weights_green = self.weights_green - alpha * (output_probs[2] - true_label_y[2])  * image_x
            self.weights_yellow = self.weights_yellow - alpha * (output_probs[3] - true_label_y[3])  * image_x


            if step % 10000 == 0:
                # test on testing sample
                # random_index = np.random.randint(len(self.independent_test_data))
                # test_sample = self.independent_test_data[random_index] 
                # test_image_x = BombDiagram.get_flat_image(test_sample)
                # test_true_label_y = np.array(BombDiagram.get_wire_to_cut(test_sample).value)
                # test_output_probs = self.multiclass_classification(test_image_x, self.weights_red, self.weights_blue, self.weights_green, self.weights_yellow)
                # test_loss = self.categorical_cross_entropy_loss(test_true_label_y, test_output_probs)

                training_loss.append(self.calc_loss_on_samples())
                testing_loss.append(self.calc_loss_on_independent_data())
                print("Step: ", step," traing loss: ", training_loss[-1], " testing loss: ", testing_loss[-1])

        # graph
        training_loss.append(self.calc_loss_on_samples())
        testing_loss.append(self.calc_loss_on_independent_data())
        print("Step: ", step," traing loss: ", training_loss[-1], " testing loss: ", testing_loss[-1])
        self.graph_loss(training_loss, testing_loss)


    def graph_loss (self, training_loss, testing_loss):
        # Plotting the losses
        plt.plot(training_loss, label='Learning Loss')
        plt.plot(testing_loss, label='Testing Loss')

        # Adding labels and title
        plt.xlabel('Number of Steps')
        plt.ylabel('Loss')
        plt.title('Learning and Testing Loss')

        # Adding legend
        plt.legend()

        # Displaying the plot
        plt.show()
                

