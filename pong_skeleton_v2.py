import gym
import numpy as np

# NOTE: This version of the skeleton code is only for the forward progogation section of the whole algorithm.
# If you complete this, you should get a program where the paddle moves randomly --> bc of random weights
# feel free to implement the backprop if you're comfortable

# Also note that some of the stuff in the completed code took us a while to figure out, so don't be afraid to ask
# anything.

# Also if you are getting an issue that ends in "OSError: [WinError 126] The specified module could not be found",
# we have not found a fix for this, but this code definitely works on ITAP computers

# If you're stuck or would like hints, either ask one of us, or feel free to look at this guy's github:
# https://github.com/dhruvp/atari-pong/blob/master/me_pong.py
# Note that this program doesn't completely follow this github, but SOME of the functionality is based off of it



# TODO: Hyperparameters, feel free to add any you think you might need
batch_size =            # TODO, Number of games to play before updating weights
hidden_layer_size =     # TODO, The hidden layer will eventually be an mx1 size matrix where m == hidden_layer_size

num_frames_in_game =    # Number of frames to allow the ai to play for (a full game usually less than 1500)

# Initializations
env = gym.make('Pong-v0')   # Setting up the pong game environment

# The main method, controls the flow of the program
def main():
    # TODO: Initialize two numpy arrays representing the two weight matrices, should be initially completely random
    #       and between -1 and 1. Make sure to get dimensions right.
    weights_one = # TODO
    weights_two = # TODO
    
    for g in range(batch_size):     # Plays batch_size games before updating weights
        print("Reseting environment\n")
        env.reset()                 # Reset environment for new game

        for frame in range(num_frames_in_game):       # Continues until # frames have run
            
            env.render()    # Display the pong screen

            proc_input = processScreen(observation)  # Process the screen and save
            # Note that the very first frame of the game is a different color than all of the rest

            # Get the action and probability of moving up using the processed input and weight matrices:
            (action, prob) = getAction(proc_input, weights_one, weights_two)
            
            observation, reward, done, info = env.step(action)  # Get relevant information, useful gym statistics
            # It should also be mentioned that the "env.step(action)" portion of this line actually indicates
            # what direction to move the paddle in the given frame
            
            if done:    # The game has ended
                break   # Break out of game loop
        
        print("Game " + str(g+1) + " finished after " + str(frame) + " frames")

    print("Closing environment")
    env.close()

def processScreen(np_array):
    # TODO: convert 210 x 160 x 3 numpy array representing the screen with RGB values into nx1 processed array
    return processed_screen

def getAction(proc_input, weights_one, weights_two):
    # Compute hidden layer, remember to activate with sigmoid or relu function (Hint: use numpy functions)
    hidden = # TODO
    # Compute output layer, remember to activate with sigmoid or relu function (Hint: use numpy functions)
    output = # TODO
    # Note: We had trouble with output from this function and noticed the other pong AI project's feed-forward
    #       section was done using the relu function to activate hidden layer values and sigmoid to activate the
    #       output.

    prob = output[0]
    
    if prob < .5:       # If the probability of moving up is less than 50%
        return (3, prob)    # 3 corresponds to moving down in gym
    else:
        return (2, prob)    # 2 corresponds to moving up in gym

# Sigmoid function: takes in an iterable and returns an iterable with the sigmoid function applied to all values
@np.vectorize
def sigmoid(x):
    return 1.0/(1+np.exp(-1*x))

# Relu function: takes in an iterable and returns an iterable with relu function applied to all values
def relu(vector):       # Don't know why you need this, but it's better
    vector[vector < 0] = 0
    return vector

if __name__ == "__main__":
    main()

