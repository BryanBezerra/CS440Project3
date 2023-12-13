from FirstTaskTrainer import FirstTaskTrainer

trainer = FirstTaskTrainer()
trainer.add_samples(5000)
trainer.train_model_stochastic(200000, .035, 10000, True)