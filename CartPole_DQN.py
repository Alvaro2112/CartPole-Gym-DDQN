import gym
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from matplotlib import pyplot as plt
import tensorflow as tf
import time



# CARTPOLE GAME SETTINGS
OBSERVATION_SPACE_DIMS = 4
ACTION_SPACE = [0,1]

# AGENT/NETWORK HYPERPARAMETERS
EPSILON_INITIAL = 1 # exploration rate
EPSILON_DECAY = 0.9427935559203139
EPSILON_MIN = 0.03627109780242037
ALPHA = 0.023057169655664934 # learning rate
ALPHA_DECAY = 0.011163645947286423 # learning rate decay
GAMMA = 0.9593081233219476 # discount factor
EXPERIENCE_REPLAY_BATCH_SIZE = 128
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = EXPERIENCE_REPLAY_BATCH_SIZE



def create_dqn():
    nn = Sequential()
    nn.add(Dense(36, input_dim=OBSERVATION_SPACE_DIMS, activation='tanh'))
    nn.add(Dense(28, activation='relu'))
    nn.add(Dense(len(ACTION_SPACE), activation='linear'))
    nn.compile(loss='mse', optimizer=Adam(lr=ALPHA, decay=ALPHA_DECAY))
    return nn
                  
                  
class DoubleDQNAgent(object):

    def __init__(self):
        self.memory = deque(maxlen=50000)
        self.online_network = create_dqn()
        self.target_network = create_dqn()
        self.epsilon = EPSILON_INITIAL
        self.has_talked = False
    
    
    def act(self, env, state):
        if np.random.rand(1) < self.epsilon :
            # explore
            return env.action_space.sample()
        else:
            # exploit
            q_values = self.online_network.predict(state)
            return np.argmax(q_values)


    def experience_replay(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        X, Y = [], []

        for state, action, reward, next_state, done in minibatch:

            y_i = self.online_network.predict(state) #experience_new_q_values

            # using online network to SELECT action
            action_index = np.argmax(self.online_network.predict(next_state)[0]) #online_net_selected_action

            # using target network to EVALUATE action
            target_net_evaluated_q_value = self.target_network.predict(next_state)[0][action_index] * np.invert(done)
            y_i[0][action]  = reward + GAMMA * target_net_evaluated_q_value

            X.append(state[0])
            Y.append(y_i[0])

        self.online_network.fit(np.array(X), np.array(Y), batch_size=len(X), verbose=0)
        
        
    def update_target_network(self):
        self.target_network.set_weights(self.online_network.get_weights())


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
                                  
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def reset_epsilon(self, episode, e_reset, reset_decay=0.85):
            #this function resets epsilon to a decreased max epsilon value using a counter
        if episode is 0: self.reset_counter = 0
        if episode % e_reset is 0 and episode > 0:
            self.reset_counter += 1
            self.epsilon = 1.0 * reset_decay**self.reset_counter

    def _reshape_state_for_net(self, state):
        return np.reshape(state,(1, OBSERVATION_SPACE_DIMS))  

def movingaverage(r_list, n):
    r_mean = []
    for i in range(n//2):
        r_mean.append(0)
    for i in range(len(r_list) - n):
        r_mean.append( sum(r_list[i:i+n])/n)
    return r_mean



def test_agent():
    env = gym.make('CartPole-v0')
    env.seed(1)
    MAX_TRAINING_EPISODES = 500
    MAX_STEPS_PER_EPISODE = 200


    agent = DoubleDQNAgent()
    trial_episode_scores = []
    epsilon_list = []
    reward_list = []
    completed = False
    time_to_win = 0

    with tf.Session() as sess:
        from keras import backend as K
        K.set_session(sess)
        agent.update_target_network()

        for episode_index in range(MAX_TRAINING_EPISODES):
                state = env.reset()
                state = np.reshape(state, [1, state.size])

                episode_score = 0

                for _ in range(MAX_STEPS_PER_EPISODE):
                    action = agent.act(env , state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [1, state.size])
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state

                    episode_score += reward

                    if done:
                        break

                if len(agent.memory) > EXPERIENCE_REPLAY_BATCH_SIZE:
                        agent.experience_replay()

                agent.update_epsilon()
                agent.reset_epsilon(episode_index, 40, 0.825)

                if episode_index % 5 == 0 :                    
                        agent.update_target_network()

                
                
                reward_list.append(episode_score)
                epsilon_list.append(agent.epsilon)
                trial_episode_scores.append(episode_score)

                if episode_index > 100:
                    r_list_smooth_100 = movingaverage(reward_list, 100)
                    for k in range(len(r_list_smooth_100)):
                        if r_list_smooth_100[k] >= 195:
                            print('The Agent has solved the enviroment')
                            completed = True


                last_100_avg = np.mean(trial_episode_scores[-100:])
                #print('E %d scored %d, avg %.2f' % (episode_index, episode_score, last_100_avg))
                if completed:
                    print('Trial 1 solved in %d episodes!' % ((episode_index - 100)))
                    time_to_win = episode_index - 100
                    break
    if time_to_win == 0:
        print('Trial not solved')
        time_to_win = 500

    return np.array(trial_episode_scores) , np.array(epsilon_list),np.array(reward_list), time_to_win


def plot_individual_trial(trial, e,r):
    plt.figure(1)
    plt.title('Double DQN CartPole v-0 Steps in Select Trial')
    plt.plot(trial)
    plt.ylabel('Steps in Episode')
    plt.xlabel('Episode')

    plt.figure(2)
    plt.title('Smoothed end Score Per Episode')
    #still need to add a offset to the moving average so it doesn't cut the front of the list off
    r_list_smooth = movingaverage(r, 10)
    r_list_smooth2 = movingaverage(r, 40)
    plt.plot(r_list_smooth)
    plt.plot(r_list_smooth2)

    plt.figure(3)
    plt.title('Greedy epsilon value')
    plt.plot(e)

    plt.show()

if __name__ == '__main__':

    total = 0
    for i in range(5):
        
        trials, e ,r , tt = test_agent()
        plot_individual_trial(trials,e,r)
        total += tt/5 

    print(total)
