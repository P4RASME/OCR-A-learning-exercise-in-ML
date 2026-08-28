import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
np.random.seed(42)

# From what I know, first i need to set up the input. 

data = pd.read_csv(r"C:\Users\ypara\OneDrive\Desktop\Documents\GitHub\ocr-for-personal-use\MNIST Digits.csv")

# The first column is the label, then column pixel 0, pixel 1, etc. We need to turn the columns into rows for matrix multiplication. 

# first, we convert the file into an array.

d_tensor = np.array(data)

d_tensor_T = d_tensor.T  # transposed tensor
Input_Tensor_train = d_tensor_T[1:, :30000] / 255.0

Result_Tensor_train = d_tensor_T[0, :30000]


Input_Tensor_dev = d_tensor_T[1:,30000:]/255

Result_Tensor_dev = d_tensor_T[0,30000:]

# These are the results against which we check our answers. 

def Relu(x):
    return np.maximum(0, x)
# This function is flat to start with then turns into y = x 

def init_params(X, neurons):
    W1 = np.random.randn(neurons, len(X)) * np.sqrt(2. / len(X))  # for matrix multiplication, rows in matrix 1 must have same length as columns in matrix 2  
    b1 = np.zeros((neurons, 1))
    W2 = np.random.randn(neurons,neurons) * np.sqrt(2./neurons) 
    b2 = np.zeros((neurons,1))
    W3 = np.random.randn(10, neurons) * np.sqrt(2. / neurons)       
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3




def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True) 
#flattens all the values into something between -0.5 and 0.5

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Relu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3
# this turns input to output

def one_hot(Y): 
    one_hot_Y = np.zeros((Y.size,Y.max() + 1 ))
    one_hot_Y[np.arange(Y.size), Y] = 1 
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
# This turns the result tensor into the intended output, with one index being one and the rest being zero to calculate the loss

def deriv_Relu(Z): 
    return Z > 0
#derivative of relu, with equals 1 (associated with True like any nonzero number) when x > 0 

def back_prop(Z1, A1, Z2, A2, A3, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1/Y.size * dZ3.dot(A2.T)
    db3 = 1/Y.size * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * deriv_Relu(Z2)
    dW2 = 1/Y.size * dZ2.dot(A1.T)
    db2 = 1/Y.size * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_Relu(Z1)
    dW1 = 1/Y.size * dZ1.dot(X.T)
    db1 = 1/Y.size * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3
# caulculating derivativres .  must divide by Y.size so that the weights don't blow up, plus it is also convention. 
def update_params( W1, b1, W2, b2 ,W3,b3, dW1,db1, dW2, db2,dW3,db3, alpha): 
    W1 = W1 - alpha * dW1 
    b1 = b1 - alpha * db1 
    W2 = W2 - alpha * dW2 
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3 
    b3 = b3 - alpha * db3 
    return W1, b1, W2, b2, W3, b3

def time_based(alpha_0,d, iteration):
    alpha = alpha_0/(1 + d * iteration)
    return alpha

def get_predictions(A3): 
    return np.argmax(A3, 0)
# finding the highest probability index from the result
def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.sum(predictions == Y)/Y.size 
# calculating the portion of results that were correct. 

def gradient_descent(X,Y,iterations, alpha_0, neurons): 
    W1, b1, W2, b2, W3, b3 = init_params(X, neurons)
    d = alpha_0/(iterations)
    n_samples = X.shape[1]
    for i in range(iterations):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        X_shuffled = X[:, indices]
        Y_shuffled = Y[indices]

        np.array_equal(X_shuffled[:, 0], X[:, indices[0]])
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_shuffled)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, A3, W2, W3, X_shuffled, Y_shuffled)
        alpha = time_based(alpha_0,d,i)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3,dW1, db1, dW2, db2, dW3, db3,alpha)
        if (i % 10 == 0): 
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y_shuffled))

    return W1, b1, W2, b2,W3,b3, alpha 

neuron_arr = [
    128,
    64,
    32,
    16
]
results = []
for neurons in neuron_arr:
    shuffle_idx = np.random.permutation(Input_Tensor_train.shape[1])
    X_train_shuffled = Input_Tensor_train[:, shuffle_idx]
    Y_train_shuffled = Result_Tensor_train[shuffle_idx]

    W1, b1, W2, b2, W3, b3, alpha = gradient_descent(
        Input_Tensor_train,
        Result_Tensor_train,
        1000,
        0.7,
        neurons
    )
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, Input_Tensor_dev)
    dev_accuracy = get_accuracy(get_predictions(A3), Result_Tensor_dev)
    results.append({"neurons": neurons, "dev_accuracy": dev_accuracy})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(r"C:\Users\ypara\OneDrive\Desktop\Documents\GitHub\ocr-for-personal-use\Complexity_optimisation_results_shuffling.csv", index=False)
print(results_df)