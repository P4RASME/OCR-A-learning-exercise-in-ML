# Dev Log 

Script-by-Script explanations of project details, my thought process and findings. 

## 1_node_setup.py 

This is the initial single hidden layer, 128 neuron OCR model for digit detection inspired by 3Blue1Brown and Samson Zhang.  This script turns an array of numbers, represented by tensors with pixel-by-pixel 
brightness descriptors of the brightness of each pixel in a 28x28 image into an array of digits (0-9) assigned to each 784 pixel row. 

### Setup Details
- Uses the first 20,000 rows of the MNIST database. 
- Has a constant learning rate of 0.5. 
- Does not use a seed generator, so the results are not perfectly deterministic. 

### Results 
This script achieved an accuracy of ~87%.  This is quite low, and could definitely be improved with further optimisation of the learning rate, which is  
explored later. 

## own_digits_1_node.py 

To explore how image processing works 
