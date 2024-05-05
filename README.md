# PLANT DETECTION MODEL
## ABSTRACT

The aim of this project is to detect the plant based on a leaf image using CNN. Looking at the current situation many do not have an idea of what plant this particular leaf belongs to. This model eases the process of the recognition of the plant. The recognized leaf can also be used in herbal ways and use it for health and medicinal purposes too. This model will definitely save the valuable time of the user by the faster recognition of the leaf. It will play a vital role in day-to-day life of a user. It's going to make a difference as it uses extensive machine learning.
## WORKFLOW
>>1. The code starts by importing necessary libraries such as "os", "numpy", "tensorflow", "tkinter","matplotlib.pyplot".
>>2. "Image Data Generator" objects are created for training and testing data. These generator apply various transformations like rescaling, shearing, zooming, and horizontal flipping to augment the training dataset.
>>3. Training and testing data is loaded using the "flow_from_directory" method specifying the target size, batch size.
>>4. The model is trained using the "fit " method on the training data.
>>5. A list of class labels is defined for later use in predicting and displaying results.
>>6. A function "predict_image" is defined for predicting the class of selected image.

