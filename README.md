# Optimized-MNIST



In this repository I have built a model for implementing a multi digit classification on the given MNIST database.

For this I decided to use a deep CNN model . I have explained the detailed architecture of this CNN later . In order to get the maximum accuracy possible I have also implemented hyperparameter tuning on this CNN using Optuna package.


In order to optimize the accuray I have implemented a deep CNN mentioned in this research paper :[https://arxiv.org/abs/2008.10400](https://arxiv.org/abs/2008.10400)

Here is a brief introduction to the model proposed in the paper: This network models consist of multiple convolution layers(10) and 2 fully connected layer at the end. In each convolution layer, a 2D convolution is performed, followed by a 2D batch normalization and ReLU activation. Max pooling or average pooling is not used after convolution. Instead, the size of feature map is reduced after each convolution because padding is not used.I have used a 3×3 kernel,so the width and height of the image is reduced by two after each convolution layer.

In order to avoid the problem of overfitting I have also added a dropout layers before each fully connected layer. Dropout rates are calculated using hyperparameter tuning.For this we are using a Optuna package. In Optuna, the goal is to minimize/maximize the objective function, which takes as input a set of hyperparameters and returns a validation score. For each hyperparameter, we consider a different range of values.



In "inference.ipynb" notebook, I'll load the MNIST multidigit classification model I created in the previous notebook with PyTorch. Then we will use this model for predicting the class of a single input image.

For implementing this notebook, please download the `main.py` and `checkpoint.pt` files that I have added in this repo.
You can download the `checkpoint.pt` from this link : 
[https://drive.google.com/file/d/1UmrntW11CtGeGkJt6KCis3yQhFUaT77j/view?usp=share_link](https://drive.google.com/file/d/1UmrntW11CtGeGkJt6KCis3yQhFUaT77j/view?usp=share_link)






