from FirstTaskTrainer import FirstTaskTrainer

trainer = FirstTaskTrainer()
trainer.add_samples(5000)
trainer.train_model_stochastic(10000000, .04)
