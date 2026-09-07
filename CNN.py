import numpy as np 
from scipy import signal
np.random.seed(24)
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
    def init(self,input_shape,kernel_size,depth):    # input_shape is a tuple containing the dimensions of the input, kernel_size represents the size of each matrix in each kernel, depth is number of kernels aka the depth of the output.
        input_depth,input_height,input_width = input_shape
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
    

