from SecondTaskTrainer import SecondTaskTrainer


a = SecondTaskTrainer()
a.add_generated_samples(50000)
a.stochastic_gradient_descent(5000000, 0.035, 10000)
