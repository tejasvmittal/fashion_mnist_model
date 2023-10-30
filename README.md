This application trains the convolution neural network to identify 10 classes of handwritten digits (0-9). 

Data: The data for this application is imported from tensorflow.keras.datasets.mnist which contains 70,000 images of handwritten digits from 10 labels. Please check out https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md to learn more about the dataset. 

Program: This model extensively uses the keras library included with tensorflow to create the convolution model. 

The model compilation automatically stops before 10 epochs after the prediction accuracy of the model exceed 99.5%. Each epoch does take a while to complete and took me on average 40s to go through each. The output of the program will be included in the form a screenshot. 