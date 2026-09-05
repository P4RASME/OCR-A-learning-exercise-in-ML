import numpy as np 
from scipy import signal

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
    def init(self,input_shape,) 