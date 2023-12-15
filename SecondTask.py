from SecondTaskTrainer import SecondTaskTrainer


a = SecondTaskTrainer()
a.add_generated_samples(10000)
a.train_model_stochastic(1000000, 0.035, 0.1, 10000)
