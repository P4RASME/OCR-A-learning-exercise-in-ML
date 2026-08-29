# Dev Log 

Script-by-Script explanations of project details, my thought process and findings. 

## 1_node_setup.py 

This is the initial single hidden layer, 128 neuron OCR model for digit detection inspired by 3Blue1Brown and Samson Zhang.  This script turns an array of numbers, represented by tensors with pixel-by-pixel descriptors of the brightness of each pixel in a 28x28 image into an array of digits (0-9) assigned to each 784 pixel row. 

### Setup Details
- Uses the first 20,000 rows of the MNIST database. 
- Has a constant learning rate of 0.5. 
- Does not use a seed generator, so the results are not perfectly deterministic. 

### Results 
This script achieved an accuracy of ~87%.  This is quite low, and could definitely be improved with further optimisation of the learning rate, which is explored later. 

## own_digits_1_node.py 

To explore how image processing and labelling works, I made my own image processing script inspired by Yann LeCun's (the creator of the MNIST) image processing for the MNIST to edit my own handwritten digits. 

### Setup Details 
- The first part of the script takes a set of images from a folder with personally handwritten digits, edits them to comply with the MNIST standards and turns each image into an array of the pixel-by-pixel brightness.
- The second part of the script trains the model on the MNIST database, and returns the dev accuracy using my handwritten digits.
- uses the same single node setup as 1_node_setup.py

### Results

Achieved 80% accuracy on my handwriting, which could be greatly improved upon.  One way to improve this could be to increase the number of personal handwriting samples, as at the moment there are only 10 (one for each digit).  However, 
this would consume a lot of time as the images would need to be drawn and labelled manually.  This accuracy also suggests that there could be overfitting, as the training accuracy was above 99%, showing a great disparity in the accuracies.  The overfitting problem will be addressed later.


## Experiment_1.py

The poor accuracy on my handwriting led me to believe that the main problem was the lack of complexity in the model (this would turn out to be false), so I added another 128 neuron hidden layer as I believed that it would let the model spot more complex patterns in handwriting.  This script tests 4 learning rate schedulers with a wide range of initial learning rates to determine the best learning rate scheduler for this case. 

### Setup Details 
- Two 128 neuron hidden layers
- power based, time based, exponential and constant learning rates were used
- The results are given [here](training_results.csv).

### Results

The results are visualised [here](Experiment_1_results.png).  As shown, the time based regime has the best accuracy overall, which is further validated by it having the highest average accuracy.  It was also observed that the dev accuracy decreases substantially for dev values greater than 0.3 for all learning regimes, suggesting that the system likely diverged as the learning rate was too high to get to reach a minimum loss.  This experiment motivated further tests with the time based regime in order to find the optimal initial learning rate.  One issue that arose here was that when I did the test multiple times, my dev accuracies would be different.  This is because the weight tensors are initialised with random values, meaning that the experiment is not deterministic.  To fix this, all future scripts will have a set seed so that results are reproducible. 


## Time_based_testing.py

Inspired by the results of Experiment 1, this script compares a variety of initial starting learning rates between 0.3 - 1 (as the boundary values here produced the best results in Experiment 1).

### Setup Details
- Script initialised with np.random.seed(42)
- Two 128 neuron hidden layers
- time based regime only, with variable initial learning rate


### Results 
The results are given [here](training_results_refined.csv).  The results were not promising.  Across the entire range of values, there was miniscule variation in the dev accuracy, averaging at 96.5%.  This end accuracy is also comparable to the accuracies achieved at the constant and time based regimes in Experiment 1, suggesting that there is a limiting factor that has not yet been addressed.  After further scrutinisation, I noticed that the training accuracies for the best results would exceed 99%, but the dev accuracy would then fall off to ~96%.  This suggests that there is overfitting to the data, meaning that the model is memorising the training data rather than generalising to numbers as a whole.  The cause of this is probably the complexity of the neural network, which allows it to form enough connections to memorise the training data set of 20,000 entirely.  This should be fixed by lowering the complexity, either by reducing the number of neurons or going back to one hidden layer. 

## Complexity_optimisation.py 

In an attempt to improve dev accuracy, this script tests the accuracy of the model at different neuron numbers, from 128 down to 16.  Moreover, shuffling was implemented as a data augmentation technique to also improve accuracy. 

### Setup details 
- 30,000 training data
- Tested complexities (neuron numbers): 128, 64, 32, 16
- Only used the time based regime, with alpha = 0.7
- Two tests were done, one without shuffling and one with shuffling.

### Results
 Non shuffling results are [here](Complexity_optmisation_results_neurons.csv) and the shuffling ones are [here](Complexity_optimisation_results_shuffling.csv).
