from FirstTaskTrainer import FirstTaskTrainer

trainer = FirstTaskTrainer()
trainer.add_samples(5000)
trainer.train_model_stochastic(1000000, .04, 1000, True)