import numpy as np
import matplotlib.pyplot as plt

class Layer_Dense:

    def __init__(self, n_input, n_neurons):
        self.weights = 0.10 * np.random.randn(n_input, n_neurons)   # (rows, columns) 
        self.biases = np.zeros((1, n_neurons))      # first arg is the shape hense biase shape is (1,n_neurons)

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
        self.transpose_below_zero = transpose_below_zero

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

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_pred_clipped = y_pred_clipped
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
        self.activation.forward(inputs)
        self.output = self.activation.output        # This is the softmax confidence scores

        return self.loss.calculate(output=self.output, y=y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples



file = "basic_eng_min2.txt"
# file = "basic_eng.txt"
sentences_path = rf"C:\Users\ujasv\OneDrive\Desktop\{file}"    
sentences_list = open(sentences_path, 'r').read().splitlines()   # lines

# this trims long items to short
trim_upto = 5         # any lines about this amout of words will be trimmed out
trimmed_continous_sentences = [i for i in sentences_list if len(i.split()) < trim_upto]

# Contains same as trimmed 
trimmed_split_sentences = []
for lines in trimmed_continous_sentences:
    list_of_words = lines.split(' ')
    
    # adding space at the end of each word but the last in the line
    list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
    trimmed_split_sentences.append(list_of_words)

# trimmed_split_sentences are 4 word sentences. each sentence is a list that contains those words as strings.
trimmed_split_sentences = [i for i in trimmed_split_sentences if len(i)==4]

print('Total batches:',len(trimmed_split_sentences))
print('total inputs in each batch:',len(trimmed_split_sentences[0]))

uniq = []
[[uniq.append(y) for y in i] for i in trimmed_split_sentences]

# TODO: sort these per unique values using andrej tots{} trick and graph the: inputs stoi, to see the relation of words with sentences and get a sense of language through visual ques.
uniq_set = set(uniq)
uniq_set_sorted = sorted(list(uniq_set))
print('Total uniques', len(uniq_set_sorted))

stoi = {s:i+1 for i,s in enumerate(uniq_set_sorted)}
stoi['..'] = 0
itos = {i:s for s,i in stoi.items()}
output_classes = len(list(stoi.keys()))

# Sentences converted to stoi projections and represent as numbers
inputs = []
for split_lines in trimmed_split_sentences:
    inputs.append([stoi[y] for y in split_lines])

# Converted to numpy array for processing
inputs = np.array(inputs)

# xx is first three items and yy is the following item
xx, yy = inputs[:,:3],inputs[:,-1]





dense1_neurons = 10

dense1 = Layer_Dense(n_input=xx.shape[1],n_neurons=dense1_neurons)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(n_input=dense1_neurons, n_neurons=output_classes)

dense1.forward(inputs=X)        # deriving dot product using inputs*weights+biases
activation1.forward(dense1.output)     

dense2.forward(activation1.output)





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




# check nnfs google doc for basic understanding of sequence of processing on
    # inputs to weights to neurons Search NNbeginner.py and check photo.
print('inputs', dense1.inputs.shape)
print('dense1 weights', dense1.weights.shape)
print('dense1 output', dense1.output.shape)
print('dense2 weights', dense2.weights.shape)
print('dense2 output', dense2.output.shape)