from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import pandas as pd
from anim3d import  Plotty_animation as pa
# This Script attempts to predict if the stocks will close higher or lower given the open time.

nnfs.init()

appl_data = r"C:\Users\ujasv\OneDrive\Desktop\codes\HOOD_Dataset.csv"

df = pd.read_csv(appl_data)


# excludes 1st column from dataframe, replaces all dollar amounts to float
X = df[df.columns[3:]].replace({'\$':""}, regex=True).astype(float)
y = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)
y = y.to_frame()        # Converting series df to frame for columns
def scaleit(scalevals):
    return scalevals/np.sum(scalevals)*100

X['ScaledVolume'] = scaleit(scalevals=df['Volume'])
# X['Open'] = scaleit(scalevals=X['Open'])
X['High'] = scaleit(scalevals=X['High'])
X['Low'] = scaleit(scalevals=X['Low'])

y['Open'] = scaleit(scalevals=X['Open'])
y['Close/Last'] = scaleit(scalevals=y['Close/Last'])

# y columns: ['Close/Last', 'Open']
# X columns ['Open', 'High', 'Low', 'ScaledVolume']

X = np.array(X)
X[:,0] = X[:,0]/np.sum(X[:,0])*100      # attempt to scale the data to workable size

y = np.array(y)
y[:,[0,1]] = y[:,[1,0]]         # making first column as open, second as close/last
output_classes = y.ndim
y_classification = y.argmax(axis=1)

class Layer_Dense:
    
    def __init__(self, n_input, n_neurons):
        self.weights = 0.10 * np.random.randn(n_input, n_neurons)   # (rows, columns) 
        self.biases = np.zeros((1, n_neurons))      # first arg is the shape hense biase shape is (1,n_zero)
        
    def forward(self, inputs):
        self.inputs = inputs
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

        max_operation = np.max(inputs, axis=1, keepdims=True)
        self.max_operation = max_operation
        transpose_below_zero = inputs - max_operation
        self.transpose_below_zerp = transpose_below_zero
        exp_values = np.exp(transpose_below_zero)
        self.exp_values = exp_values
        
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

dense1_neurons = 10

# Using index X.shape[1] since thats what throws in the integer.
dense1 = Layer_Dense(n_input=X.shape[1],n_neurons=dense1_neurons)
activation1 = Activation_ReLU()

# dense2 will contain 2 classification. good or bad.
dense2 = Layer_Dense(n_input=dense1_neurons, n_neurons=output_classes)

dense1.forward(inputs=X)        # deriving dot product using inputs*weights+biases
activation1.forward(dense1.output)     

dense2.forward(activation1.output)

def pst(y_vals):
    if y_vals.ndim == 2:            # If just rows and columns then
        x = range(y_vals.shape[0])
        for i in range(y_vals.shape[1]):    # iterate over all columns.
            plt.scatter(x, y_vals[:,i])
    if y_vals.ndim == 3:    # could do <= as well
        x = range(y_vals.shape[0])
        for i in range(y_vals.shape[0]):
            for a in range(y_vals.shape[2]):
                plt.scatter(x, y_vals[i,:,a])

def plotit(y_vals):
    if y_vals.ndim == 2:
        x = range(y_vals.shape[0])
        for i in range(y_vals.shape[1]):
            plt.plot(x, y_vals[:,i])
    if y_vals.ndim == 3:    # could do <= as well
        x = range(y_vals.shape[1])
        for i in range(y_vals.shape[0]):
            for a in range(y_vals.shape[2]):
                plt.plot(x, y_vals[i,:,a])

class Loss:         # This class is inherited by Loss_Categorical_Crossentropy
    def __init__(self):
        self.y_pred_clipped = None
        self.prior_negative_log = None
        self.post_negative_log = None
        self.data_loss = None

    def calculate(self, output, y):
        sample_losses= self.forward(output, y)
        self.post_negative_log = sample_losses
        
        data_loss = np.mean(sample_losses)
        self.data_loss = data_loss

        return data_loss

class Loss_Categorical_Crossentropy(Loss):

    def forward(self, y_pred, y_true):
      
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_pred_clipped = y_pred_clipped
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                                  range(samples), 
                                  y_true
                                  ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1  
            )
        self.correct_confidences = correct_confidences
        self.prior_negative_log = correct_confidences
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

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
loss = loss_activation.forward(inputs=dense2.output, y_true=y_classification)

raw_dense1_outputs = dense1.output      # These are pure input*weigh+bias' of the 1st layer
first_layer_relu_outputs = activation1.output
raw_dense2_outputs = dense2.output      # puter inp*wei+bia of the 2nd layer
softmax_outputs = loss_activation.activation.output 
ground_truth_values = y_classification

# These are the correct confidences from the softmax given ground_truth_values
correct_confidences = loss_activation.loss.prior_negative_log
batch_loss_as_log = loss_activation.loss.post_negative_log

# the total loss of the neural network
mean_batch_loss = loss_activation.loss.data_loss
print(y.shape,
    dense1.weights.shape,
    dense1.output.shape,
    dense2.weights.shape,
    dense2.output.shape,
    softmax_outputs.shape)

predictions = np.argmax(softmax_outputs, axis=1)

# If targets are one-hot encoded - convert them
class_targets = y_classification
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions==y_classification)
print('acc:', accuracy)

lowest_loss = 99999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

count = 0

plt.ion() # enable interactive mode
fig, ax = plt.subplots()
pause = 0.1
total_iters = 5000

for iterations in range(total_iters):

    dense1.weights += 0.05 * np.random.randn(X.shape[1],dense1_neurons)
    dense1.biases += 0.05 * np.random.randn(1, dense1_neurons)
    dense2.weights += 0.05 * np.random.randn(dense1_neurons,output_classes)
    dense2.biases += 0.05 * np.random.randn(1,2)
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss = loss_activation.forward(inputs=dense2.output, y_true=y_classification)

    softmax_outputs = loss_activation.activation.output 
    correct_confidences = loss_activation.loss.prior_negative_log
    batch_loss_as_log = loss_activation.loss.post_negative_log
    mean_batch_loss = loss_activation.loss.data_loss

    predictions = np.argmax(softmax_outputs, axis=1)
    accuracy = np.mean(predictions == y_classification)
    
    # If the current loss becomes lower then copy all weights and biases
        # Else if loss was greater this time then recall the last best dense w&b
    if loss < lowest_loss:      
        
        ax.clear()  
        
        # x limit is stable so no need to set x limit since len of all rows and nth column is always same
        ax.set_xlim(159,170)      
        ax.set_ylim(0.3,0.6)
        
        # np.hstack stacks columns next to each other
        # columns are the differnt sets of data and rows are the datapoints on y axis themselves
        # transposed_d = softmax_outputs +
        data_to_plot = np.hstack((y,softmax_outputs))       # First arg itself is the column bunch hence double bracs
        ax.plot(data_to_plot)   # plotting the data
        
        # Refresh plot 
        fig.canvas.draw()   
        plt.pause(pause)
        ax.clear()  

        
        count+=1
        print('New weights. c/i=', count,iterations, 'loss/acc:', loss, '/', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

plt.show()