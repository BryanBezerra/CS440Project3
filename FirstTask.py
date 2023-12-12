from FirstTaskTrainer import FirstTaskTrainer

trainer = FirstTaskTrainer()
trainer.add_samples(2000)
trainer.train_model_stochastic(2000, .05)
