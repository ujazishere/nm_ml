import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt
nnfs.init()

# Two lists containing 4 items. Shape of this is (2,4). matrix of 2 rows and 4 columns
# np.random.rand(2,4) 
# similarly following conntains2 lists each of those lists contains 3 items,
    # and each of those items contains 4 items
# np.random.rand(2,3,4)
# np.randint(1,5,size=(2,4)) This one returns rand ints between(inclusive) 1 and 4 with shape 2,4

class Layer_Dense:

    def __init__(self, n_input, n_neurons):
        # .randn() produces random gaussian distribution as float ranging  between -0.3 and +0.3 since using 0.1 multiple
        self.weights = 0.10 * np.random.randn(n_input, n_neurons)   # (rows, columns) 
        self.biases = np.zeros((1, n_neurons))      # first arg is the shape hense biase shape is (1,n_neurons)

    def forward(self, inputs):
        self.inputs = inputs

        # The column are the neuron outputs. n_columns are n_neurons.
        # each neuron of all columns(one per column) will be a product and sum of their inputs and associated weights
        # See NNFS google doc first page for more info.
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    

class Activation_ReLU:

    def forward(self, inputs) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)     # making negative numbers 0 and retaining positive numbers as is.

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0      # search GPT
    

class Activation_Softmax:

    def forward(self, inputs):
        self.inputs = inputs

        # np.max(inputs) takes in highest number and discards rest
        # if you just do `np.max(input)` it will take all the batch inputs and give you highest of them all
        # rather, we have to use max of only the batch being used and retain dimensions
        # We are using this highest number and subtracting from inputs one at a timme, 
            # this makes the highest number 0 and all others negative esentially transposing them all below zero
        # we then exponentiatiate keeping dimensions intact for all process
            # exponentiation is e raised to the input.
        max_operation = np.max(inputs, axis=1, keepdims=True)
        self.max_operation = max_operation

        transpose_below_zero = inputs - max_operation
        self.transpose_below_zero = transpose_below_zero
        # exponentiation +ve numbers above 710 gives positive infinity uptp 300 decimals long
            # -ve direction below 710 approaches 1 with over 300 decimal places
            # two digit number either +ve or -ve will quickly blow up to 43 decimal/digit places

        exp_values = np.exp(transpose_below_zero)
        self.exp_values = exp_values

        # we then use the exponentiated value and divide it by the sum of all values 
        # in the batch of that data retaining dimension. 
        # This operation results in all values ranging between 0,1 which prevents overflow error
        # these are the confidence scores. 1 being 100% confidence and 0 being 0% confidence 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)


# ***********VVI***********
# The softmax_outputs are the confidence scores/probability distributions. between 1 and 0 
# `loss` involves taking negative log of
    # the softmax output corresponding to the ground truth values/class.
# lower the confidence in softmax output the larger the 
    # loss shown in the -ive log output in
    # loss_catagorical_crossentropy() retaining the extent of loss.
# as confidence approaces 1(100% confidence) the -ve log output(the loss_cat_xentrpy())
    # approaches 0(indicating no loss) and 
# as the softmax confidence decreases towards zero the -ve log(the loss_cat_crsentrpy()) blows up towards infinity(showing extensive loss).
# So first, those far extreme ends(0 and 1) are clipped in the softmax output to avoid runtime error using
    # np.clip making any and every 0 a tad bit bigger and any and every 1 a tad bit smaller
# we then take the correct confidences as compared with the ground truth class/values
# this gives us just one output per input batch which is the
    # confidence score of the associated ground truth class.
# we then take the -ve log of each of those values. This is the sample loss
# we then find the mean of the sample loss. Mean is just the average.
# ***********VVI***********

# Calculating loss with Categorical Cross-Entropy
# This is a Common Loss class
class Loss:         # This class is inherited by Loss_Categorical_Crossentropy
    def __init__(self):
        # clipped 0 and 1 to avoid infinity error. Takaes into account on-hot vals
        self.y_pred_clipped = None
        # These are correct confidences as compared with ground truth values
        self.prior_negative_log = None
        # -ve log done on correct confidences. These are the sample losses
        self.post_negative_log = None
        # data_loss is the singular mean of the negative log values(sample_losses)
        self.data_loss = None

    # Calculates the data regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # This gets the clipped -ve log values of the target value index confidences
        sample_losses= self.forward(output, y)
        self.post_negative_log = sample_losses
        data_loss = np.mean(sample_losses)

        self.data_loss = data_loss

        return data_loss

# Cross-Entropy Loss. Inheriting common loss class here.
class Loss_Categorical_Crossentropy(Loss):

    # Forward pass 
    def forward(self, y_pred, y_true):
      
        # Number of samples in a batch
        samples = len(y_pred)

        # Clipping data to prevent division by 0
        # Clipping both ends to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_pred_clipped = y_pred_clipped
        # Probabilities for target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                                  range(samples), 
                                  y_true
                                  ]

        # Mask values - Only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1  
            )
        self.correct_confidences = correct_confidences
        # Prior to negative log
        self.prior_negative_log = correct_confidences
        # Losses 
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Crossentropy()
    
    def forward(self, inputs, y_true):
        # Second dense layers' raw output gets fed in here to get softmax outputs
        self.activation.forward(inputs)
        self.output = self.activation.output        # This is the softmax confidence scores

        # these somftmax outputs are fed into 
        return self.loss.calculate(output=self.output, y=y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
