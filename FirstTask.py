from FirstTaskTrainer import FirstTaskTrainer

trainer = FirstTaskTrainer()
trainer.add_samples(5000)
(training_loss, testing_loss) = trainer.train_model_stochastic(10000000, .04)
trainer.graph_loss(training_loss, testing_loss)