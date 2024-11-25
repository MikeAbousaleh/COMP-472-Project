COMP 472 Project
This project compares multiple machine learning models on the CIFAR-10 dataset, including Naive Bayes, Decision Tree, MLP (Multilayer Perceptron), and VGG11.
* It is important to note that this program is unable to save and load models previously trained properly, therefore if it the program were to run it would do the whole training process from the start.

Table of Contents
Project Overview
File Descriptions
Setup and Installation
Data Preprocessing
Running the Code
Results and Evaluation

Project Overview
This project aims to explore how different machine learning models perform on the CIFAR-10 dataset. Models compared include Naive Bayes, Decision Tree, MLP, and VGG11.

File Descriptions
1.	NaiveBayes.py
Contains the implementation of the Naive Bayes classifier for image classification.
2.	DecisionTree.py
Contains the implementation of a Decision Tree classifier, built using recursive binary splitting.
3.	MLP.py
Defines the  model MLP and its training process. It includes code for training a simple neural network using PyTorch.
4.	VGG11.py
Contains the implementation of the VGG11 convolutional neural network model. The model is built using PyTorch.
5.	main.py
The main script that loads the CIFAR-10 dataset, trains models, and generates performance metrics. It includes model training, evaluation, and confusion matrix plotting.
6.	README.md
This file provides an overview of the project, detailed instructions, and descriptions of each model.

Setup and Installation
Install the required packages: pip install torch torchvision numpy scikit-learn matplotlib seaborn 

Data Preprocessing
The dataset is preprocessed in the load_cifar10_data function which does the following:
1.	Download CIFAR-10 using torchvision.datasets.CIFAR10.
2.	Resize the images to 224x224 pixels which is needed for the VGG11 input size.
3.	Normalize the images using ImageNet's mean and standard deviation values for VGG11.
4.	Flatten the images for models like Naive Bayes, Decision Tree, and MLP.
5.	Reduce the dimensionality of the images for MLP and other simpler models down to 50.

Running the Code
To train and see the results for the Naive Bayes, Decision Tree, MLP, VGG11 models, execute the main.py file. The file does:
Training Models:
•	Naive Bayes is trained on the feature vectors.
•	Decision Tree is trained on the feature vectors.
•	MLP is trained on the reduced feature vectors (50 dimensions).
•	VGG11 is trained using the raw 224x224 images.

Evaluating Models:
•	The models' performances are evaluated on test data.
•	Accuracy is calculated for each model.
•	Loss is included for each epoch of both MLP and VGG11 models.
•	Confusion matrices are plotted for each model’s predictions on the test data.
•	Classification reports (precision, recall, F1-score) are generated

The current code is meant to display the main models of the program.

Results and Evaluation
The training script will print out the following metrics for each model:
Training Accuracy: Accuracy achieved on the training data.
Testing Accuracy: Accuracy achieved on the test data.

 A confusion matrix will be displayed once the training of each model has been completed. As well,  a classification report for each model will also be generated once the program is completed.








