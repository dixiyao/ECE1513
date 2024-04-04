## Test the code
### Prepare the dataset
Download the dataset from ```https://www.kaggle.com/datasets/andrewmvd/medical-mnist``` and unzip as the folder dataset. The dataset should look like
```
--main folder
    --dataset
        --AbdomenCT
        --BreastMRI
        --ChestCT
        --CXR
        --Hand
        --HeadCT
```
Regarding the linear regression, please run the command
```
python3 linear_regression.py
```
To test the fully connected neural network
```
python3 linear_layers.py
```
## Results
### Settings
Resolution 128x128
Randomly shuffle the 58954 Images. 40000 of them are for training, 10000 are for validation, and 8954 are for testing.
### FCN
Epoch:  1 Training Loss:  0.05215028427623448 Validation Loss:  0.016390682611894566 Test Loss:  0.014698033822236955
Training Accuracy:  0.986475 Validation Accuracy:  0.9954 Test Accuracy:  0.9968729059638151
Epoch:  2 Training Loss:  0.015235951855901797 Validation Loss:  0.02578645301551785 Test Loss:  0.023309773856673825
Training Accuracy:  0.99595 Validation Accuracy:  0.9937 Test Accuracy:  0.9934107661380389
Epoch:  3 Training Loss:  0.009598002109488385 Validation Loss:  0.01733306193257688 Test Loss:  0.009652791043090029
Training Accuracy:  0.997425 Validation Accuracy:  0.9971 Test Accuracy:  0.9979897252624526
Epoch:  4 Training Loss:  0.005997115033701636 Validation Loss:  0.009577476927883416 Test Loss:  0.004242873215791518
Training Accuracy:  0.99835 Validation Accuracy:  0.9976 Test Accuracy:  0.9985481349117713
Epoch:  5 Training Loss:  0.004344222867015546 Validation Loss:  0.007758358119329032 Test Loss:  0.005391165240180996
Training Accuracy:  0.998625 Validation Accuracy:  0.9984 Test Accuracy:  0.9989948626312263
Epoch:  6 Training Loss:  0.004025670253288401 Validation Loss:  0.01159498320710712 Test Loss:  0.008941567946837535
Training Accuracy:  0.99885 Validation Accuracy:  0.9982 Test Accuracy:  0.99821308912218
Epoch:  7 Training Loss:  0.0022790966760308988 Validation Loss:  0.02030206131920127 Test Loss:  0.015855430153599557
Training Accuracy:  0.999275 Validation Accuracy:  0.9971 Test Accuracy:  0.9975429975429976
Epoch:  8 Training Loss:  0.0020625775702762888 Validation Loss:  0.011329567227141431 Test Loss:  0.0060836615331997775
Training Accuracy:  0.9993 Validation Accuracy:  0.9984 Test Accuracy:  0.9985481349117713
Epoch:  9 Training Loss:  0.00716356498429034 Validation Loss:  0.026712241403672538 Test Loss:  0.00986622829620283
Training Accuracy:  0.99845 Validation Accuracy:  0.9973 Test Accuracy:  0.997766361402725
Epoch:  10 Training Loss:  0.004072212881913758 Validation Loss:  0.008750934580451196 Test Loss:  0.007426914306626987
Training Accuracy:  0.998675 Validation Accuracy:  0.9976 Test Accuracy:  0.9972079517534063

Learning rate: 0.01 and SGD.
### Linear Regression
Linear Regression Score:  0.741797927397005
Train Accuracy:  0.7008 MSE Loss:  0.271988288994935
Test Accuracy:  0.4658253294616931 MSE Loss:  0.750862176485608
Validation Accuracy:  0.4526 MSE Loss:  0.7722789727711111