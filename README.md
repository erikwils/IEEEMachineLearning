# Machine Learning Pong
This is a tutorial/walk through for how to make a machine learning pong game. It is intended to use a single layer neural network to accomplish this. This tutorial was created by the chairs of Purdue IEEE Computer Society in Fall 2019.

## Setup
This tutorial is intended to be used with the Python 3.7 language, and the Gym OpenAI library. Install Python onto your machine in order to follow along with the rest of the tutorial.

## Starting Input Layer
To begin, all Machine Learning algoithms need some sort of an input layer. For this scenario, the screen is the input, however, you need to decide what would be the best parts of the screen to actually be used in the input vector. Whether that be the entire screen, only the position of the ball and paddles, or a sampled part of the window, your choice needs to be inputted into an Nx1 vector, where N is the input values you take from the screen.

## Deciding the Hidden Layer
The next step of the algorithm is determining the hidden layer. The hidden layer is the middle step in determining the final output. The important step here is determining the number of nodes that the hidden layer has, a value M. This number depends on the input layer and the output layer. In this scenario, there is only one output node, the probability of moving up, so we'd recommend somewhere between 2-5 nodes. Next, we'll begin to compute the values of this hidden layer.

## Initializing the Weight Matrix
Now that you have decided the number of nodes in the input layer, N, and the number of nodes in the hidden layer, M, we need to create a weight matrix that will allow us to compute the hidden layer. The weight matrix is an NxM matrix that to begin is initialized with random values between 0 and 1. The purpose of this matrix is to multiply the input vector by it in order to compute the values of the hidden layer. The machine learning aspect of the algorithm comes from updating these weights to output the hidden layer nodes to more accurately tell the final output node to move up or done. Initially, our model is untrained, so these weight values are just random to start.

## Computing the Hidden Layer and the Activation Function
To compute the actual values of the hidden layer, you must take your Nx1 input layer and multiply it by your randomly initialized NxM weight matrix to create an Mx1 hidden layer. This layer needs to be scaled down to values between 0 and 1, so one way to do this is by applying an activation function to it. We'd recommend using the sigmoid function here to scale all the values. After doing this you have successfully computed the hidden layer.

## Repeating the Process and Creating an Output
Now that you have the hidden layer, you're going to want to do the same goal over again, just using the hidden layer this time as the first layer. So create and randomly initialize a new weight matrix that is Mx1 and multiply it times the hidden layer to create a 1x1 output, the final output. 

## Training the Model
So after doing all of the aforementioned steps, you're going to have a model that is really bad, since it's been initialized randomly. So we need to train the model, which is essentially updating all the weight matricies with different values so that they produce better outputs that will allow for our pong algorithm to win more frequently. There are a few different ways to do this, and however you want to, the choice is up to you. Some of the common ways are through back propogation or through genetic modeling, among many others. Back propagation is essentially computing derivatives of the model to find the slope, and then using the derivatives to minimize the error of the function, and updating the model accordingly. The genetic model creates about 100 different random weight matrices, and has them all play against each other. You then keep the 50 best models, and then create 50 new models, have them play, keep the winners, so on and so forth until 
## Conclusions 
