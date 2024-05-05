# PLANT DETECTION MODEL

## ABSTRACT

The aim of this project is to detect the plant based on a leaf image using CNN. Looking at the current situation many do not have an idea of what plant this particular leaf belongs to. This model eases the process of the recognition of the plant. The recognized leaf can also be used in herbal ways and use it for health and medicinal purposes too. This model will definitely save the valuable time of the user by the faster recognition of the leaf. It will play a vital role in day-to-day life of a user. It's going to make a difference as it uses extensive machine learning.
## WORKFLOW
This repository contains code for a leaf classification project utilizing machine learning techniques. The project aims to classify leaves into various plant classes using convolutional neural networks (CNNs). Below is an outline of the key components and functionalities of the code:

1. Importing Necessary Libraries
The code begins by importing essential libraries such as os, numpy, tensorflow, tkinter, and matplotlib.pyplot. These libraries are crucial for tasks such as handling directories, numerical computations, deep learning operations, GUI development, and data visualization.

2. Image Data Augmentation
To enhance the robustness of the model and mitigate overfitting, "Image Data Generator" objects are created for both training and testing data. These generators apply various transformations like rescaling, shearing, zooming, and horizontal flipping to augment the training dataset. Augmentation helps in increasing the diversity of the training data, thereby improving the model's generalization ability.

3. Loading Training and Testing Data
The training and testing data are loaded using the flow_from_directory method, specifying parameters such as the target size of images and batch size. This method facilitates the seamless loading of data directly from directories, making it convenient to work with large datasets.

4. Model Training
The model is trained using the fit method on the training data. During training, the model learns to classify leaves into different plant classes by adjusting its parameters based on the provided training examples and corresponding labels.

5. Class Labels Definition
A list of class labels is defined for later use in predicting and displaying results. These labels correspond to the different plant classes present in the dataset and are essential for interpreting the model's predictions.

6. Prediction Function
A function named predict_image is defined to predict the class of a selected image. This function takes an input image, preprocesses it, and passes it through the trained model to obtain the predicted class label. The predicted label can then be used for further analysis or visualization.

## DATASET
The dataset is a comprehensive collection of leaf images representing a wide range of plant classes. Each image in the dataset corresponds to a leaf from various plant species, offering a rich resource for research and development in leaf classification and plant recognition tasks.

Input:- The input for plant type detection is provided by selecting an image through a graphical user interface (GUI) using the "Select Image" button, which opens a file dialog for the user to choose an image file

Output:- The output is obtained by passing the selected image through the trained Convolutional Neural Network (CNN) model, which predicts the plant type and its corresponding label, showcasing the result on the graphical user interface (GUI)..

