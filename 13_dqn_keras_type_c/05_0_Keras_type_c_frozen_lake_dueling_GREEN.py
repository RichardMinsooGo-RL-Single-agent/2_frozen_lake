import random
import numpy as np
import time, datetime
from collections import deque
import pylab
import sys
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import *
from keras.models import Sequential,Model
import keras
from keras import backend as K_back
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

state_size = 64
action_size = 5
n_rows = 9
n_cols = 9

model_path = "save_model/"
graph_path = "save_graph/"
    
# Make folder for save data
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

load_model = True

class Frozen_Lake:
    def __init__(self):
        # player velocity, max velocity, downward accleration, accleration on flap
        self.agent_row = 0
        self.agent_col = 0
        self.rand_init = np.random.randint(low=0, high=3)
        
    def reset_env(self):
        
        self.agent_row = 0
        self.agent_col = 0
        self.rand_init = np.random.randint(low=0, high=3)
        
        state = np.zeros((n_rows,n_cols))

        # rand_init = 0
        state[2][(self.rand_init+1)%9] = 2
        state[4][(self.rand_init+4)%9] = 2
        state[6][(self.rand_init+7)%9] = 2
        state[8][8] = 4
        state[0][0] = 5
        
        return state
        
    def frame_step(self, action, ep_step):

        if action == 0:
            if (self.agent_row + 1) < 9:
                self.agent_row += 1
        if action == 1:
            if self.agent_row > 0:
                self.agent_row -= 1
        if action == 2:
            if self.agent_col > 0:
                self.agent_col -= 1
        if action == 3:
            if (self.agent_col+1) < 9:
                self.agent_col += 1

        agent_pos = np.zeros((9,9))
        agent_pos[self.agent_row][self.agent_col] = 5

        ice_lake = np.zeros((9,9))
        hole_1_col = int((self.rand_init+ep_step+1)%9)
        hole_2_col = int((self.rand_init+ep_step+4)%9)
        hole_3_col = int((self.rand_init+ep_step+7)%9)
        ice_lake[2][hole_1_col] = 2
        ice_lake[4][hole_2_col] = 2
        ice_lake[6][hole_3_col] = 2
        ice_lake[8][8] = 4

        next_state = agent_pos + ice_lake
        # print(next_state)

        # reward = agent_row - 8 + agent_col - 8
        reward = -1
        
        done = False

        if np.count_nonzero(next_state == 7) > 0:
            if ep_step < 15:
                reward = reward - 200
            else:
                reward = reward - 100
            # done = True

        if np.count_nonzero(next_state == 9) > 0:
            done = True
            reward = 500
        if ep_step == 500:
            done = True
            
        return next_state, reward, done

# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN_agent:
    def __init__(self):

        # get size of state and action
        self.progress = " "
        self.state_size = state_size
        self.action_size = action_size
        
        # train time define
        self.training_time = 20*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.2
        # final value of epsilon
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.0005
        self.epsilon = self.epsilon_max
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 500
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 64
        self.input_shape = (n_rows,n_cols,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        
        state = Input(shape=self.input_shape)        
        
        net1 = Convolution2D(32, kernel_size=(3, 3),activation='relu', \
                             padding = 'valid', input_shape=self.input_shape)(state)
        net2 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding = 'valid')(net1)
        net3 = MaxPooling2D(pool_size=(2, 2))(net2)
        net4 = Flatten()(net3)
        lay_2 = Dense(units=self.hidden2,activation='relu',kernel_initializer='he_uniform',\
                  name='hidden_layer_1')(net4)
        value_= Dense(units=1,activation='linear',kernel_initializer='he_uniform',\
                      name='Value_func')(lay_2)
        ac_activation = Dense(units=self.action_size,activation='linear',\
                              kernel_initializer='he_uniform',name='action')(lay_2)
        
        #Compute average of advantage function
        avg_ac_activation = Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
        
        #Concatenate value function to add it to the advantage function
        concat_value = Concatenate(axis=-1,name='concat_0')([value_,value_])
        concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])

        for i in range(1,self.action_size-1):
            concat_value = Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
            concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])

        #Subtract concatenated average advantage tensor with original advantage function
        ac_activation = Subtract()([ac_activation,concat_avg_ac])
        
        #Add the two (Value Function and modified advantage function)
        merged_layers = Add(name='final_layer')([concat_value,ac_activation])
        model = Model(inputs = state,outputs=merged_layers)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.zeros((self.batch_size, n_rows, n_cols, 1))
        next_states = np.zeros((self.batch_size, n_rows, n_cols, 1))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        tgt_q_value_next = self.target_model.predict(next_states)
        
        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(tgt_q_value_next[i]))
                
        # Decrease epsilon while training
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else :
            self.epsilon = self.epsilon_min
            
        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        # choose an action_arr epsilon greedily
        action_arr = np.zeros(self.action_size)
        action = 0
        
        if random.random() < self.epsilon:
            # print("----------Random action_arr----------")
            action = random.randrange(self.action_size)
            action_arr[action] = 1
        else:
            # Predict the reward value based on the given state
            Q_value = self.model.predict(state)
            action = np.argmax(Q_value[0])
            action_arr[action] = 1
            
        return action_arr, action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        # Save the variables to disk.
        self.model.save_weights(model_path+"/model.h5")
        save_object = (self.epsilon, self.episode, self.step)
        with open(model_path + '/epsilon_episode.pickle', 'wb') as ggg:
            pickle.dump(save_object, ggg)

        print("\n Model saved in file: %s" % model_path)

def main():
    
    agent = DQN_agent()
    game = Frozen_Lake()
    
    # Initialize variables
    # Load the file if the saved file exists
    if os.path.isfile(model_path+"/Model_dueling_0.h5"):
        agent.model.load_weights(model_path+"/Model_dueling_0.h5")
        if os.path.isfile(model_path + '/epsilon_episode.pickle'):
            
            with open(model_path + '/epsilon_episode.pickle', 'rb') as ggg:
                agent.epsilon, agent.episode, agent.step = pickle.load(ggg)
            
        print('\n\n Variables are restored!')

    else:
        print('\n\n Variables are initialized!')
        agent.epsilon = agent.epsilon_max
    
    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    # print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    # initialize target model
    agent.Copy_Weights()
    
    while time.time() - start_time < agent.training_time and avg_score < 470:
    # while agent.episode < 1:
        
        state = game.reset_env()
        done = False
        agent.score = 0
        ep_step = 0
        rewards = 0
        
        # if agent.progress == "Training":
        #     print(state)
        state = state.reshape(1,n_rows,n_cols,1)
        
        while not done and ep_step < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"            
            else:
                agent.progress = "Training"

            ep_step += 1
            agent.step += 1
            
            action_arr, action = agent.get_action(state)
            
            next_state, reward, done = game.frame_step(action, ep_step)
            
            rewards += reward
            
            next_state = next_state.reshape(1,n_rows,n_cols,1)
            
            agent.append_sample(state, action, reward, next_state, done)
            
            # print("next_state_shape :\n", next_state.shape)
            state = next_state
            
            # sys.exit()
            
            if agent.progress == "Training":
                agent.train_model()
                if done or ep_step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()

            agent.score = rewards
                    
            if done:
                if agent.progress == "Training":
                    agent.episode += 1
                    scores.append(agent.score)
                    episodes.append(agent.episode)
                    avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>6,d}'.format(agent.episode),'/ ep step :{:>5,d}'.format(ep_step), \
                      '/ rewards :{:>4.1f}'.format(rewards),'/ status :', agent.progress, \
                      '/ epsilon :{:>1.4f}'.format(agent.epsilon),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break
    # Save model
    agent.save_model()
    
    pylab.plot(episodes, scores, 'b')
    pylab.savefig("./save_graph/cartpole_duelingdqn.png")

    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()

if __name__ == "__main__":
    main()
