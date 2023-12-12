import math

import numpy as np
import matplotlib.pyplot as plt
import random
from BombDiagram import BombDiagram


class FirstTaskTrainer:
    NUM_COLORS = 4

    def __init__(self, diagram_size=20):
        self.diagram_size = diagram_size
        self.weights = self.generate_starting_weights()
        self.samples = []
        self.independent_test_data = []
        self.used_samples = []
        self.loss_values = []
        self.num_success = 0

    def generate_starting_weights(self):
        num_features = self.diagram_size * self.diagram_size * self.NUM_COLORS + 1
        start_magnitude = (num_features ** 0.5) ** -1
        return np.random.uniform(-start_magnitude, start_magnitude, num_features)

    def add_samples(self, num_samples):
        for i in range(num_samples):
            self.samples.append(BombDiagram(self.diagram_size))
            self.independent_test_data.append(BombDiagram(self.diagram_size))

    def clear_samples(self):
        self.samples.clear()

    def sigmoid(self, bomb_diagram):
        input_vec = bomb_diagram.get_flat_image()
        return 1 / (1 + math.e ** (-np.dot(input_vec, self.weights)))

    def predict(self, bomb_diagram):
        predicted_value = self.sigmoid(bomb_diagram)
        if predicted_value < 0.5:
            return 0
        else:
            return 1

    def loss(self, bomb_diagram):
        predicted_val = self.sigmoid(bomb_diagram)
        observed_val = bomb_diagram.is_dangerous()
        return -observed_val * math.log(predicted_val) - (1 - observed_val) * math.log(1 - predicted_val)

    def calc_loss_on_samples(self):
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
        loss_sum = 0
        for sample in self.independent_test_data:
            loss_sum += self.loss(sample)
        return loss_sum / len(self.independent_test_data)

    def train_model_stochastic(self, num_steps, alpha):
        training_loss = []
        testing_loss = []
        for i in range(num_steps):
            if len(self.samples) == 0:
                self.samples = self.used_samples.copy()
                random.shuffle(self.samples)
                self.used_samples.clear()

            data_point = self.samples.pop()
            self.used_samples.append(data_point)
            prediction = self.predict(data_point)
            self.weights = self.weights - alpha * (prediction - data_point.is_dangerous()) * data_point.get_flat_image()
            if i % 1000 == 0 or i == 1:
                print("Step", i, "Loss:", self.calc_loss_on_samples(), "Loss on independent data:", self.calc_loss_on_independent_data())
                training_loss.append(self.calc_loss_on_samples())
                testing_loss.append(self.calc_loss_on_independent_data())
                
        print("Step", num_steps, "Loss:", self.calc_loss_on_samples(), "Loss on independent data:", self.calc_loss_on_independent_data())
        training_loss.append(self.calc_loss_on_samples())
        testing_loss.append(self.calc_loss_on_independent_data())
        return training_loss, testing_loss
    
    def graph_loss(self, training_loss, testing_loss):
        plt.plot(training_loss, label='Learning Loss')
        plt.plot(testing_loss, label='Testing Loss')
        plt.xlabel('Number of Steps')
        plt.ylabel('Loss')
        plt.title('Learning and Testing Loss')
        plt.legend()
        plt.show()

