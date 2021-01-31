# KineticNeuralNetwork

## Introduction
This is a Kinetic Neural Network built from scratch in C++. Not only function properly, it was also well optimized to minimize training time. The network is able to read both CSV and common image file as input. You can tweak the parameters to adjust input/output size or training performance.

## Parameter
Parameter | Description 
--- | --- 
NETWORK_DEPTH | Number of layers in the network 
NETWORK_STRUCTURE | Array of neuron amount of each layer 
TRANSFER_FUNCTION_TYPE | Array of transfer function of each layer (0 for sigmoid, 1 for relu)
LEARNING_RATE | Array of learning rate of each layer (0 for sigmoid, 1 for relu)
TRAINING_EXTRACT_SIZE | Number of samples to be extracted from CSV training dataset
TESTING_EXTRACT_SIZE | Number of samples to be extracted from CSV testing dataset
TRAIN_ITER | Amount of backpropagation to be done
TEST_ITER | Amount of sample tested during each test
TEST_PERIOD | Program will run a test for every TEST_PERIOD of iter
BATCH_SIZE | Batch size
outputIsLabel | Set true if label is an integer ranging (0-no. class), set false if label is binary array with size of no. class 

![alt text](https://github.com/jason2468087/KineticNeuralNetwork/blob/main/Result/KNN%20acc_time.png?raw=true)

![alt text](https://github.com/jason2468087/KineticNeuralNetwork/blob/main/Result/KNN%20Plot.png?raw=true)
