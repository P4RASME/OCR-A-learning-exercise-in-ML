import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time
start_time = time.time()
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

def init_params(X):
    W1 = np.random.randn(128, len(X)) * np.sqrt(2. / len(X))  # for matrix multiplication, rows in matrix 1 must have same length as columns in matrix 2  
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(128,128) * np.sqrt(2./128) 
    b2 = np.zeros((128,1))
    W3 = np.random.randn(10, 128) * np.sqrt(2. / 128)       
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
# Updating weights and biases. 
def get_predictions(A3): 
    return np.argmax(A3, 0)
# finding the highest probability index from the result
def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.sum(predictions == Y)/Y.size 
# calculating the portion of results that were correct. 
def time_based(alpha_0,d, iteration):
    alpha = alpha_0/(1 + d * iteration)
    return alpha
# time bsased scheduling for learning rate 
def power(alpha_0, iteration, s, p):
    alpha = alpha_0/(1 + iteration/s)**p
    return alpha
# power shceduling for learning rate. 
def exp(alpha_0, iteration, s):
    alpha = alpha_0 * 0.95**(iteration/s)
    return alpha
# exponential scheduling for learning rate, similar to power scheduling. 

#def gradient_descent(X,Y,iterations, alpha_0): 
#    W1, b1, W2, b2, W3, b3 = init_params(X)
#    d = alpha_0/(iterations)
#    for i in range(iterations):
#        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
#        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, A3, W2, W3, X, Y)
 #       alpha = time_based(alpha_0,d,i)
#        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3,dW1, db1, dW2, db2, dW3, db3,alpha)
#        if (i % 10 == 0): 
#            print("Iteration: ", i)
#            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
#
#    return W1, b1, W2, b2,W3,b3, alpha 


def make_constant_scheduler():
    def scheduler(alpha_0, iteration):
        return alpha_0
    return scheduler
# keeps the value constant 
def make_time_based_scheduler(iteration):
    def scheduler(alpha_0, iteration):
        d = alpha_0 / iterations
        return time_based(alpha_0, d, iteration)
    return scheduler

def make_power_scheduler(s, p):
    def scheduler(alpha_0, iteration):
        return power(alpha_0, iteration, s, p)
    return scheduler

def make_exp_scheduler(s):
    def scheduler(alpha_0, iteration):
        return exp(alpha_0, iteration, s)
    return scheduler

def gradient_descent_generic(X, Y, iterations, alpha_0, scheduler):
    W1, b1, W2, b2, W3, b3 = init_params(X)
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, A3, W2, W3, X, Y)
        alpha = scheduler(alpha_0, i)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Alpha: ", alpha)
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
    return W1, b1, W2, b2, W3, b3, alpha


learning_rates = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1,
    3
    ]

accuracies = []


power_s = 150
power_p = 1
exp_s = 30
iterations = 1000

schedulers = {
    "constant": make_constant_scheduler(),
    "time_based": make_time_based_scheduler(iterations),
    "power": make_power_scheduler(power_s, power_p),
    "exp": make_exp_scheduler(exp_s)
}

colors = {
    "constant": "blue",
    "time_based": "green",
    "power": "red",
    "exp": "purple"
}

results = []

fig2, ax2 = plt.subplots()

output_dir = r"C:\Users\ypara\OneDrive\Desktop\Documents\GitHub\ocr-for-personal-use"

for scheduler_name, scheduler in schedulers.items():
    accuracies = []
    for num in learning_rates:
        W1, b1, W2, b2, W3, b3, alpha = gradient_descent_generic(
            Input_Tensor_train,
            Result_Tensor_train,
            iterations,
            num,
            scheduler
        )

        _, _, _, _, _, A3_dev = forward_prop(
            W1, b1, W2, b2, W3, b3,
            Input_Tensor_dev
        )

        dev_predictions = get_predictions(A3_dev)
        dev_accuracy = get_accuracy(dev_predictions, Result_Tensor_dev)
        accuracy_percent = dev_accuracy * 100
        accuracies.append(accuracy_percent)

        weights_filename = f"weights_{scheduler_name}_{num}.npz"
        np.savez(fr"{output_dir}\{weights_filename}", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


        results.append({
            "scheduler": scheduler_name,
            "learning_rate": num,
            "dev_accuracy": accuracy_percent,
            "weights_file": weights_filename
        })

        print(f"Scheduler: {scheduler_name} | Alpha: {num} | Dev Accuracy: {accuracy_percent:.2f}%")

    ax2.plot(learning_rates, accuracies, marker="o", color=colors[scheduler_name], label=scheduler_name)

ax2.set_xscale("log")
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Dev Accuracy (%)")
ax2.set_title("Learning Rate vs Dev Accuracy by Scheduler")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(results)


results_df.to_csv(fr"{output_dir}\training_results.csv", index=False)

end_time = time.time()
elapsed = end_time - start_time
print(f"Time elapsed: {elapsed:.2f} seconds")