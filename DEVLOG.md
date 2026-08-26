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
- The results are given in [training_results.csv](training_results.csv).

### Results

The results are visualised [here](Experiment_1_results.png).  As shown, the time based regime has the best accuracy overall, which is further validated by it having the highest average accuracy. TBC

