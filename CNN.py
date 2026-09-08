


# set params
n_samples = 30000 
l_rate = 0.001
iterations = 500








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

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
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
    def forward(self,input):
        self.input = input 
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j],self.kernels[i,j],"valid")
        return self.output
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape) # initialises kernel gradients, with all zeroes of course
        input_gradient = np.zeros(self.input_shape) #initialises the input gradient

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i,j] = signal.correlate2d(self.input[j],output_gradient[i],"valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i,j], "full") # specifying a full convolution here!
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient 
        return input_gradient 

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape 
    def forward(self,input): 
        return np.reshape(input,self.output_shape) # gives the input the output shape 
    
    def backward(self,output_gradient,learning_rate):
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

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Softmax(Layer):
    def forward(self, input):
        self.input = input
        exp_values = np.exp(input - np.max(input))
        self.output = exp_values / np.sum(exp_values)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # when combined with categorical cross-entropy, the gradient simplifies to just passing through
        return output_gradient


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient) 
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient
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

def train(network, loss, loss_prime, x_train, y_train, epochs, learning_rate): 
    for epoch in range(epochs): 
        error = 0 
        correct = 0
        for i in range(len(x_train)):
            output = x_train[i]
            for layer in network: 
                output = layer.forward(output)

            error += loss(y_train[i], output)
            
            # calculate accuracy
            prediction = get_predictions(output)
            if prediction == np.argmax(y_train[i]):
                correct += 1

            grad = loss_prime(y_train[i], output)
            for layer in reversed(network):
                grad = layer.backward(grad,learning_rate)

        error /= len(x_train)  # divides the error by the number of inputs 
        accuracy = (correct / len(x_train)) * 100  # calculate percentage accuracy

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
X_train, Y_train = prepare_data(Input_Tensor_train, Result_Tensor_train,n_samples  )

# prepare dev set
X_dev, Y_dev = prepare_data(Input_Tensor_dev, Result_Tensor_dev, len(Result_Tensor_dev))

# train the network
network = train(network, categorical_cross_entropy, categorical_cross_entropy_prime, 
      X_train, Y_train, iterations, l_rate)

# evaluate on dev set
dev_accuracy = evaluate_dev(network, X_dev, Y_dev)
print(f"\nDev Accuracy: {dev_accuracy:.2f}%")

end_time = time.time()
elapsed = end_time - start_time
print(f"Time elapsed: {elapsed:.2f} seconds")
