# OCR- A learning exercise in ML
Developing and exploring an Optical Character Recognition Model as a way to learn Machine Learning fundamentals.

## First Steps
Inspired by [3Blue1Brown's](https://youtu.be/aircAruvnKk?si=RFFeUgjbC_ldnth5) neural networks course, I made a simple 1 node digit detector in Python with primarily the use of numpy (no Tensorflow/Pytorch).  Developed 
this model by adding a variable learning rate after testing the efficacy of various ways in which the learning rate can be varied.  I then added another node in an attempt to improve the accuracy of the 
system, which proved to not be very effective, at least so far.

After this more 'exploratory' phase, I feel that I should design a methodology and maybe make a more formal document to record my findings as I attempt to expand and make this model more accurate.  More detailed explanations at DEVLOG.md. 

## Project Milestones

- Created a basic single node model that uses the MNIST database with a constant learning rate, inspired by 3Blue1Brown and Samson Zhang's video on building a neural network from scratch.
- Explored various simple learning rate schedulers (exponential, time based, and power scheduling) and tested their relative effectiveness.
- Added a second hidden layer so that the model can capture more complex patterns.
- Inspired by LeCun et. al's image formatting process for the MNIST database, set up an image processing script for my own handwritten numbers that seamlessly converts handwritten digits into compatible arrays.
- Optmised the time based scheduling method, alongside shuffling batches to achieve 97.28% accuracy on the MNIST.
- Created a Convolutional Neural Network from scratch with stochastic gradient descent, outperforming the previous model on all tests done so far.  

## Rough notes 

### Optimising for accuracy 

Model Details: 
- Two Hidden Layers with 128 neurons each
- 20,000 datapoints for testing
- possible learning rate schedulers: constant, time-based, power, and exponential

The first experiment involved testing a variety of initial learning rates (alpha_0) with a variety of learning rate schedulers in order to assess which scheduler had the best performance.  It turned out that time-based scheduling performed better than any other, which led me to assess the performance of various initial learning rates with time-based scheduling.  At this point, I had reached a little bit of a road-block, realising that all the dev accuracies were stuck around 96%, even though the training accuracy would often be 99%+.  This likely means that my model is overfitting to the data, likely due to the large number of neurons and the two hidden layers and my relatively small training dataset of 20,000 points, leading the model to memorise the training data instead of developing the flexibility needed to tackle the validation data.

Once I have accounted for this overfitting, I will re-do the experiments with the time-based approach to see if I can break the 96% barrier. 

The ideal alpha_0 lies between 0.7-0.8 for the time_based tests. 


### An attempt to reduce overfitting

- Reducing the number of neurons in each layer only reduced the accuracy further, as seen in Complexity_optimisation_results_neurons.csv
- The next step will be implementing data augmentation, which essentially involves shuffling the dataset at each iteration to make it seem 'new'.
- Shuffling the dataset had marginal impact, in fact reducing average accuracy by 1%
- We may have reached a limit for this model

## CNN Implementation 
Filters are grids that matrix multiply across the entire image- they are basically the analog for weights.  The reason its called 'convolutional' is that these filters act as filter functions, as seen in convolutions from the fourier course, they take the 28x28 pixel image and reduce it in size.  The number of filters you have correspond to the number of images your produce- ie. 

1 image + 2 filters = 2 filtered images

The key forward propagation process is: 

input layer --> filtering + activation function ---> max pooling ---> filtering  + activation function---> max pooling ---> output. 

We are doing a very similar matrix multiplication, just with more dimensions. 

filter has associated bias, exact same thing as before.  The key step is still Z = activation_function(XW + B)

what are your questions: 

- What makes this better than a normal neural network? 
uses fewer paramters as the same, small filter is just slid across the whole image. less parameters ---> overfitting which was a problem earlier. 
Identifies FEATURES, not precise mathematical positions, so is not vulnerable to rotations and such. 
less parameters also means less memory and faster. 

- Why do they make images blurrier? 
To identify general features rather than hone in on fine details, less overfitting and more general use case.  The max pooling takes an 'average' of a given space, only letting the model see the broad strokes.

- Kernels are like weights, but they don’t do the same job.  Kernels in the convolutional layer are changed and refined to create the best filters for the actual weights and biases, that are found in the DENSE or ‘fully connected’ layers. 

- These layers do the actual learning, the convolutional layers, as reported earlier, just exploit the 2d structure to help pick out the best features for the dense layers to be the most accurate. 

So, the script I am making is a 3 layer structure, but only two layers are doing ‘actual’ learning, the third is just picking out features by optimising filter parameters. 

