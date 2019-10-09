import numpy as np
import gym
import random
import time

class PongPlayer:
    batch_size = 1          # Number of games to base score on
    term_frame = 2700       # Number of frames to play before terminating game (if not already finished)
    
    def __init__(self, weights_one, weights_two, play_full):
        self.weights_one = weights_one
        self.weights_two = weights_two
        self.score = 0
        self.play_full = play_full
    
    def playGames(self):
        for game in range(self.batch_size):
            observation = env.reset()

            for frame in range(self.term_frame):
                if self.play_full:
                    env.render()    # DEFINITELY remove this for actual implementation

                proc_input = self.screenProcess(observation)
                (action, prob) = self.getAction(proc_input)

                observation, reward, done, info = env.step(action)

                if done or (not self.play_full and reward < 0):
                    self.score = frame
                    break

    def screenProcess(self, np_array):
        np_array = np_array[34:194]
        np_array = np_array[::2, ::2, 0].flatten()

        np_array[(np_array != 109) & (np_array != 144)] = 1
        np_array[np_array != 1] = 0
        return np_array

    def getAction(self, input_layer):
        hidden_unactivated = np.dot(self.weights_one, input_layer)
        hidden = relu(hidden_unactivated)

        output_unactivated = np.dot(self.weights_two, hidden)
        output = sigmoid(output_unactivated)

        if output[0] < .5:
            return (3, output[0])
        else:
            return (2, output[0])

    def mutateWeights(self):    # NOTE: POssibly consider using map() instead, would be WAY simpler and possibly faster
        percent = 0.05  # Percent of weights to mutate. NOTE: Possibly decrease (.01 or .001)
        temp_w1 = self.weights_one.flatten()
        temp_w2 = self.weights_two.flatten()
        
        rand_ind1 = np.random.choice(temp_w1.size, size=int(temp_w1.size * percent))
        rand_ind2 = np.random.choice(temp_w2.size, size=int(temp_w2.size * percent))

        #temp_w1[rand_ind1] = np.random.randn(rand_ind1.size)
        #temp_w2[rand_ind2] = np.random.randn(rand_ind2.size)

        temp_w1[rand_ind1] += np.random.normal(0, 0.1, rand_ind1.size)
        temp_w2[rand_ind2] += np.random.normal(0, 0.1, rand_ind2.size)

        self.weights_one = temp_w1.reshape((200, 6400))
        self.weights_two = temp_w2.reshape((1, 200))

    #def crossover(self, oth

    def copy(self):
        return PongPlayer(self.weights_one.copy(), self.weights_two.copy(), False) # .copy() here is important

        


input_size = 6400           # Size of processed screen matrix
hidden_layer_size = 200     # Hidden layer size
number_players = 500
env = gym.make('Pong-v0')

@np.vectorize
def sigmoid(x):
    return 1.0/(1+np.exp(-1*x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def runGA():
    number_generations = 15
    pong_players = [0] * number_players

    for p in range(number_players):
        pong_players[p] = PongPlayer(np.random.randn(hidden_layer_size, input_size), np.random.randn(1, hidden_layer_size), False)
    print("Initialized first generation of players.")

    for g in range(number_generations):
        print("\nRunning new generation " + str(g+1) + "\n")
        pong_players = runGeneration(pong_players)

    best_player = pong_players[0]
    for p in pong_players:
        if (best_player.score < p.score):
            best_player = p
    print("Best score of last generation: " + str(best_player.score))
    time.sleep(2)
    best_player.play_full = True
    best_player.playGames()

    return best_player


def runGeneration(players):
    new_players = []
    
    max_score = 0
    count = 0
    for p in players:
        count = count + 1
        p.playGames()
        print("playing game" + str(count))
        if (max_score < p.score):
            max_score = p.score
    print("max score: " + str(max_score))
    
    for p in players:
        if p.score > random.randint(0, max_score):
            temp = p.copy()
            temp.mutateWeights()
            new_players.append(temp)
    while len(new_players) < len(players):
        p = random.choice(players)
        temp = p.copy()
        temp.mutateWeights()
        new_players.append(temp)
    return new_players
    

final_player = runGA()
