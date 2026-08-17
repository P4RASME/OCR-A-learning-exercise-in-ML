import numpy as np 
import pandas as pd 

# From what I know, first i need to set up the input. 

data = pd.read_csv(r"C:\Users\ypara\OneDrive\Desktop\OCR_proj\MNIST Digits.csv")

# The first column is the label, then column pixel 0, pixel 1, etc. We need to turn the columns into rows for matrix multiplication. 

# first, we convert the file into an array.

d_tensor = np.array(data)

d_tensor_T = d_tensor.T  # transposed tensor
Input_Tensor_train = d_tensor_T[1:, :20000] / 255.0

Result_Tensor_train = d_tensor_T[0, :20000]


Input_Tensor_dev = d_tensor_T[1:,20000:]/255

Result_Tensor_dev = d_tensor_T[0,20000:]

# These are the results against which we check our answers. 

def Relu(x):
    return np.maximum(0, x)


def init_params(X):
    W1 = np.random.randn(128, len(X)) * np.sqrt(2. / len(X))  
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(10, 128) * np.sqrt(2. / 128)       
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2




def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True) 

def forward_prop(W1, b1,W2, b2, X ):
    Z1 = W1.dot(X) + b1 
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2 
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2 

def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size,Y.max() + 1 ))
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_Relu(Z): 
    return Z > 0

def back_prop(Z1,A1,A2,W2,X, Y): 
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/Y.size * dZ2.dot(A1.T)
    db2 = 1/Y.size * np.sum(dZ2, 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * deriv_Relu(Z1)
    dW1 = 1/Y.size * dZ1.dot(X.T)
    db1 = 1/Y.size * np.sum(dZ1, 1, keepdims = True)
    return dW1, db1, dW2, db2 

def update_params(W1, b1, W2, b2 , dW1,db1, dW2, db2, alpha): 
    W1 = W1 - alpha * dW1 
    b1 = b1 - alpha * db1 
    W2 = W2 - alpha * dW2 
    b2 = b2 - alpha * db2 
    return W1, b1, W2, b2 

def get_predictions(A2): 
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.sum(predictions == Y)/Y.size 

def power(alpha_0, iteration, s, p):  # s means step, p means power. 
    alpha = alpha_0/(1 + iteration/s)^p
    return alpha
 
def exp(alpha_0, iteration, s):
    alpha = alpha_0 * 0.93**(iteration/s)
    return alpha

def time_based(alpha_0,d, iteration):
    alpha = alpha_0/(1 + d * iteration)
    return alpha


def gradient_descent(X,Y,iterations, alpha_0): 
    W1, b1, W2, b2 = init_params(X)
    d = alpha_0/(iterations)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1,A1,A2,W2,X, Y)
        alpha = time_based(alpha_0,d,i)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0): 
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2, alpha 




W1, b1, W2, b2,alpha = gradient_descent(Input_Tensor_train, Result_Tensor_train,1000, 0.5)
print(alpha)

_, _, _, A2_dev = forward_prop(W1, b1, W2, b2, Input_Tensor_dev)

 #Get predictions and calculate accuracy
 
dev_predictions = get_predictions(A2_dev)
dev_accuracy = get_accuracy(dev_predictions, Result_Tensor_dev)

print(f"Final Dev Set Accuracy: {dev_accuracy * 100:.2f}%")
