from SecondTaskTrainer import SecondTaskTrainer


a = SecondTaskTrainer()
a.add_generated_samples(5000)
a.train_model_stochastic(1000000, 0.035, 0, 2000)
