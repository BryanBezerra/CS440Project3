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

# − ∑_c=1 I{y = c} ln p_c
def categorical_cross_entropy_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_predicted)) / len(y_true)
    return loss

# probabilities : [0.23451, 0.2414556, 0.13134, 0.25516]
def multiclass_classification(input_vector, weights_red, weights_blue, weights_green, weights_yellow):
    dot_product_red = np.dot(input_vector, weights_red)
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
    output = np.array([softmax_red, softmax_blue, softmax_green, softmax_yellow])

    return output

def sgd_update(weights, gradients, alpha):
    weights -= alpha * gradients
    return weights

def training_model_sgd(x, true_label, weights_red, weights_blue, weights_green, weights_yellow, alpha, num_steps):
    for step in range(num_steps):
        # forward pass
        output_probs = multiclass_classification(x, weights_red, weights_blue, weights_green, weights_yellow)

        loss = categorical_cross_entropy_loss(true_label, output_probs)

        weights_red = sgd_update(weights_red, output_probs[0] - true_label[0], alpha)
        weights_blue = sgd_update(weights_blue, output_probs[1] - true_label[1], alpha)
        weights_green = sgd_update(weights_green, output_probs[2] - true_label[2], alpha)
        weights_yellow = sgd_update(weights_yellow, output_probs[3] - true_label[3], alpha)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

    # just return the final weights of each color
    return weights_red, weights_blue, weights_green, weights_yellow
