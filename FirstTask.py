from FirstTaskTrainer import FirstTaskTrainer

trainer = FirstTaskTrainer()
trainer.add_generated_samples(2000)
trainer.train_model_stochastic(480000, .035, 0, 1200, True)

trainer = FirstTaskTrainer()
trainer.add_generated_samples(2500)
trainer.train_model_stochastic(800000, .035, 0, 2000, True)

trainer = FirstTaskTrainer()
trainer.add_generated_samples(3000)
trainer.train_model_stochastic(1200000, .035, 0, 3000, True)
#
trainer = FirstTaskTrainer()
trainer.add_generated_samples(5000)
trainer.train_model_stochastic(2000000, .035, 0, 10000, True)

trainer = FirstTaskTrainer()
trainer.add_generated_samples(50000)
trainer.train_model_stochastic(4000000, .035, 0, 20000, True)
