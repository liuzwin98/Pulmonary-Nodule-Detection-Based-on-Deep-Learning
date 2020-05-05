from procedures.trainer import *

print("Training CT-GAN Injector...")
CTGAN_inj = Trainer(isInjector=True)
CTGAN_inj.train(epochs=70, batch_size=16, sample_interval=50)  # origin 200 epochs
print('Done.')