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
# Gradient descent

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
