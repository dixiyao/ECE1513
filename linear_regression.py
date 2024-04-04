import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

# Function to load and preprocess images from a subfolder
def load_and_preprocess_images(main_folder_path, subfolders, target_size=(128, 128)):
    images = []
    labels = []

    for class_label, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(main_folder_path, subfolder)

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, target_size)

            image = image / 255.0

            images.append(image)
            labels.append(class_label)

    return np.array(images), np.array(labels)

main_folder_path = "./dataset"
subfolders = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]

all_images, all_labels = load_and_preprocess_images(main_folder_path, subfolders)

print(all_images.shape, all_labels.shape)
sequence=np.arange(all_images.shape[0])
np.random.shuffle(sequence)
all_images=all_images[sequence]
all_labels=all_labels[sequence]
data_train=all_images[:40000]
labels_train=all_labels[:40000]
data_valid=all_images[40000:50000]
labels_valid=all_labels[40000:50000]
data_test=all_images[50000:]
labels_test=all_labels[50000:]

# Linear regression
model=LinearRegression()
model.fit(data_train.reshape(data_train.shape[0], -1), labels_train)
train_reg_score=model.score(data_test.reshape(data_test.shape[0], -1), labels_test)
print("Linear Regression Score: ", train_reg_score)
predict_train_label=model.predict(data_train.reshape(data_train.shape[0], -1))
mse_loss_train=np.mean((predict_train_label-labels_train)**2)
predict_train_label=np.round(predict_train_label)
accuracy=np.sum(predict_train_label==labels_train)/len(labels_train)
print("Train Accuracy: ", accuracy,"MSE Loss: ",mse_loss_train)
predict_test_label=model.predict(data_test.reshape(data_test.shape[0], -1))
mse_loss_test=np.mean((predict_test_label-labels_test)**2)  
predict_test_label=np.round(predict_test_label)
accuracy=np.sum(predict_test_label==labels_test)/len(labels_test)
print("Test Accuracy: ", accuracy,"MSE Loss: ",mse_loss_test)
predict_valid_label=model.predict(data_valid.reshape(data_valid.shape[0], -1))
mse_loss_valid=np.mean((predict_valid_label-labels_valid)**2)
predict_valid_label=np.round(predict_valid_label)
accuracy=np.sum(predict_valid_label==labels_valid)/len(labels_valid)
print("Validation Accuracy: ", accuracy,"MSE Loss: ",mse_loss_valid)