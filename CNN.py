# set params
n_samples = 30000 
l_rate = 0.001
iterations = 500
b_size = 32

import numpy as np 
from scipy import signal
import pandas as pd
import time
np.random.seed(24)

start_time = time.time()

# initial data stuff: 

data = pd.read_csv(r"C:\Users\ypara\OneDrive\Desktop\Documents\GitHub\ocr-for-personal-use\MNIST Digits.csv")

# The first column is the label, then column pixel 0, pixel 1, etc. We need to turn the columns into rows for matrix multiplication. 

# first, we convert the file into an array.

d_tensor = np.array(data)

d_tensor_T = d_tensor.T  # transposed tensor
Input_Tensor_train = d_tensor_T[1:, :30000] / 255.0

Result_Tensor_train = d_tensor_T[0, :30000]


Input_Tensor_dev = d_tensor_T[1:,30000:]/255

Result_Tensor_dev = d_tensor_T[0,30000:]

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate=None):
        # TODO: update parameters and return input gradient
        pass
    
    def update_parameters(self, learning_rate, batch_size):
        # Default implementation for layers without parameters (Activation, Reshape, etc.)
        pass
# Before starting, we define a new datatype, layer, so that we can perform the entire process by simply running through a python loop. 
# The layer contains a forward pass function, backward pass function and variables to store the input and output. 


class Convolutional(Layer):
    def __init__(self,input_shape,kernel_size,depth):    # input_shape is a tuple containing the dimensions of the input, kernel_size represents the size of each matrix in each kernel, depth is number of kernels aka the depth of the output.
        input_depth,input_height,input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape 
        self.input_depth = input_depth 
        self.output_shape = (depth, input_height - kernel_size + 1,input_width - kernel_size + 1 )
        self.kernels_shape = (depth, input_depth,kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape) # do not have to compute the biases as the biases have the same shape as the output, as in normal neural networks.
        
        # Mini-batch gradient accumulators
        self.kernels_gradient_acc = np.zeros(self.kernels_shape)
        self.biases_gradient_acc = np.zeros(self.output_shape)
        
    def forward(self,input):
        self.input = input 
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j],self.kernels[i,j],"valid")
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        kernels_gradient = np.zeros(self.kernels_shape) # initialises kernel gradients, with all zeroes of course
        input_gradient = np.zeros(self.input_shape) #initialises the input gradient

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i,j] = signal.correlate2d(self.input[j],output_gradient[i],"valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], "full") # specifying a full convolution here!
        
        # Accumulate gradients instead of updating immediately
        self.kernels_gradient_acc += kernels_gradient
        self.biases_gradient_acc += output_gradient
        
        return input_gradient
    
    def update_parameters(self, learning_rate, batch_size):
        # Update parameters using the average gradient of the batch
        self.kernels -= learning_rate * (self.kernels_gradient_acc / batch_size)
        self.biases -= learning_rate * (self.biases_gradient_acc / batch_size)
        
        # Reset accumulators for the next batch
        self.kernels_gradient_acc.fill(0.0)
        self.biases_gradient_acc.fill(0.0)

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape 
    def forward(self,input): 
        return np.reshape(input,self.output_shape) # gives the input the output shape 
    
    def backward(self,output_gradient, learning_rate=None):
        return np.reshape(output_gradient,self.input_shape) # gives the output gradient the input shape, allowiing for backprop


def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # defined an epsilon to remove any log(0) or log(negative) values
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))   # this ist he categorical cross entropy function

def categorical_cross_entropy_prime(y_true, y_pred): #  this is the output gradient, the derivative of the CCE function combined with softmax
    return y_pred - y_true  # simplified gradient when using softmax + categorical cross entropy together

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate=None):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Softmax(Layer):
    def forward(self, input):
        self.input = input
        exp_values = np.exp(input - np.max(input))
        self.output = exp_values / np.sum(exp_values)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        # when combined with categorical cross-entropy, the gradient simplifies to just passing through
        return output_gradient


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        
        # Mini-batch gradient accumulators
        self.weights_gradient_acc = np.zeros((output_size, input_size))
        self.biases_gradient_acc = np.zeros((output_size, 1))
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    def backward(self, output_gradient, learning_rate=None):
        self.weights_gradient_acc += np.dot(output_gradient, self.input.T)
        self.biases_gradient_acc += output_gradient
        
        input_gradient = np.dot(self.weights.T, output_gradient) 
        return input_gradient
        
    def update_parameters(self, learning_rate, batch_size):
        # Update weights using the average gradient of the batch
        self.weights -= learning_rate * (self.weights_gradient_acc / batch_size)
        self.biases -= learning_rate * (self.biases_gradient_acc / batch_size)
        
        # Reset accumulators for the next batch
        self.weights_gradient_acc.fill(0.0)
        self.biases_gradient_acc.fill(0.0)
    # defining a dense layer removes the need for having to make separate weights and copy paste individual calculations, as it allows us to just add a layer whenever we please

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
# we need another activation that is not softmax as softmax resets the weights to probabilities adding up to one, not positive and negative weights that we would need for learning. 
def predict(network,input): 
    output = input 
    for layer in network:
        output = layer.forward(output)
    return output # this is simply the whole forward prop process

def get_predictions(output):
    return np.argmax(output) # finding the highest probability index from the result

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size # calculating the portion of results that were correct.

def train(network, loss, loss_prime, x_train, y_train, epochs, learning_rate, batch_size): 
    num_samples = len(x_train)
    
    for epoch in range(epochs): 
        error = 0 
        correct = 0
        
        # Shuffle data at the beginning of every epoch for optimal mini-batching
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Loop through data in steps of batch_size
        for b in range(0, num_samples, batch_size):
            batch_indices = indices[b:b + batch_size]
            current_batch_size = len(batch_indices) # Accounts for the final smaller batch
            
            # Step 1: Accumulate gradients over the mini-batch
            for idx in batch_indices:
                output = x_train[idx]
                for layer in network: 
                    output = layer.forward(output)

                error += loss(y_train[idx], output)
                
                prediction = get_predictions(output)
                if prediction == np.argmax(y_train[idx]):
                    correct += 1

                grad = loss_prime(y_train[idx], output)
                for layer in reversed(network):
                    grad = layer.backward(grad)
            
            # Step 2: Trigger parameter updates at the end of the batch
            for layer in network:
                layer.update_parameters(learning_rate, current_batch_size)

        error /= num_samples  
        accuracy = (correct / num_samples) * 100  

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {error:.4f}, Accuracy: {accuracy:.2f}%")
    
    return network

def evaluate_dev(network, x_dev, y_dev):
    correct = 0
    for i in range(len(x_dev)):
        output = predict(network, x_dev[i])
        prediction = get_predictions(output)
        if prediction == np.argmax(y_dev[i]):
            correct += 1
    
    dev_accuracy = (correct / len(x_dev)) * 100
    return dev_accuracy

network = [
    Convolutional((1,28,28),kernel_size= 3, depth=5),
    Tanh(),
    Reshape((5,26,26),(5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Tanh(), 
    Dense(100,10),
    Softmax()
]

def prepare_data(input_tensor, result_tensor, num_samples):
    X = []
    Y = []
    for i in range(num_samples):
        x = input_tensor[:, i].reshape(1, 28, 28)
        y = np.zeros((10, 1))
        y[int(result_tensor[i])] = 1
        X.append(x)
        Y.append(y)
    return X, Y # preparing data so that it is in a 2d shape rather than a list, and one hot encoding as softmax demands it. 

# prepare training samples 
X_train, Y_train = prepare_data(Input_Tensor_train, Result_Tensor_train, n_samples)

# prepare dev set
X_dev, Y_dev = prepare_data(Input_Tensor_dev, Result_Tensor_dev, len(Result_Tensor_dev))

# train the network
network = train(network, categorical_cross_entropy, categorical_cross_entropy_prime, 
      X_train, Y_train, iterations, l_rate, b_size)

# evaluate on dev set
dev_accuracy = evaluate_dev(network, X_dev, Y_dev)
print(f"\nDev Accuracy: {dev_accuracy:.2f}%")

end_time = time.time()
elapsed = end_time - start_time
print(f"Time elapsed: {elapsed:.2f} seconds")
