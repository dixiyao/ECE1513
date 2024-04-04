import matplotlib.pyplot as plt
import numpy as np

traning_loss=np.load('training_loss.npy')
validation_loss=np.load('validation_loss.npy')
test_loss=np.load('test_loss.npy')
plt.plot(range(traning_loss.shape[0]),traning_loss,label='Training Loss')
plt.plot(range(validation_loss.shape[0]),validation_loss,label='Validation Loss')
plt.plot(range(test_loss.shape[0]),test_loss,label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')