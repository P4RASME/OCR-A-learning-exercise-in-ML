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

## Basic logs/notes 

### Optimising for accuracy 

Model Details: 
- Two Hidden Layers with 128 neurons each
- 20,000 datapoints for testing
- possible learning rate schedulers: constant, time-based, power, and exponential

The first experiment involved testing a variety of initial learning rates (alpha_0) with a variety of learning rate schedulers in order to assess which scheduler had the best performance.  It turned out that time-based scheduling performed better than any other, which led me to assess the performance of various initial learning rates with time-based scheduling.  At this point, I had reached a little bit of a road-block, realising that all the dev accuracies were stuck around 96%, even though the training accuracy would often be 99%+.  This likely means that my model is overfitting to the data, likely due to the large number of neurons and the two hidden layers and my relatively small training dataset of 20,000 points, leading the model to memorise the training data instead of developing the flexibility needed to tackle the validation data.

Once I have accounted for this overfitting, I will re-do the experiments with the time-based approach to see if I can break the 96% barrier. 

The ideal alpha_0 lies between 0.7-0.8 for the time_based tests. 

