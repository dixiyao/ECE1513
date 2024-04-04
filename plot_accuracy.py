import matplotlib.pyplot as plt
import numpy as np

training_accuracy=np.load('training_accuracy.npy')
validation_accuracy=np.load('validation_accuracy.npy')
test_accuracy=np.load('test_accuracy.npy')
plt.plot(range(training_accuracy.shape[0]),training_accuracy,label='Training Accuracy')
plt.plot(range(validation_accuracy.shape[0]),validation_accuracy,label='Validation Accuracy')
plt.plot(range(test_accuracy.shape[0]),test_accuracy,label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')