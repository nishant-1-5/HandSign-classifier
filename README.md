# Sign-Language Classifier using Neural Network
This project involves building a sign language classification model using a neural network. The model is trained on a [dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) containing images of hand gestures representing various signs from American Sign Language (ASL).
## Dependencies
<ul><li>Matplotlib</li>
<li>Numpy</li>
<li>Tensorflow</li>
  <li>pandas</li>
</ul>
You can install these packages via pip.

## Files
-   `sign_mnist_train.csv`: Training dataset containing image data and labels.
-   `sign_mnist_test.csv`: Test dataset containing image data and labels.
-   `handSign`: Main script to train and evaluate the sign language classifier model.

## Description
This project uses a neural network to classify input digits between 0 and 9, It consists of the following layers:-
1. **Input Layer**:  784 neurons, corresponding to the flattened 28x28 pixel images.
2. **Hidden Layer 1**: The first hidden layer has 30 neurons and uses the ReLU activation function.
3.  **Hidden Layer 2**: The second hidden layer has 20 neurons and uses the ReLU activation function.
4. **Hidden Layer 3**: The third hidden layer has 15 neurons and uses the ReLU activation function.
5. **Output Layer**: The output layer has 25 neurons (one for each symbol) and uses a linear activation function. 

# Result
<img src="/img/Pred.png" />


