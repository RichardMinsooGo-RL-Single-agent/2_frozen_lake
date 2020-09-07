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
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        
        # These are hyper parameters for the DQN_agent
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.step = 0
        self.score = 0
        self.episode = 0
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 500
        
        self.input_shape = (n_rows,n_cols,1)
        
        # create main model and target model
        self.model = self.build_model()
        
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

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        # choose an action_arr epsilon greedily
        action_arr = np.zeros(self.action_size)
        
        Q_value = self.model.predict(state)
        action = np.argmax(Q_value[0])
        action_arr[action] = 1
            
        return action_arr, action

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
def main():
    
    agent = DQN_agent()
    game = Frozen_Lake()
    
    if load_model:
        agent.model.load_weights(model_path + "/Model_dueling_0.h5")
    
    avg_score = 0
    episodes, scores = [], []
    
    # start training    
    # Step 3.2: run the game
    display_time = datetime.datetime.now()
    # print("\n\n",game_name, "-game start at :",display_time,"\n")
    start_time = time.time()
    
    while agent.episode < 20:
        
        state = game.reset_env()
        done = False
        agent.score = 0
        ep_step = 0
        rewards = 0
        
        state = state.reshape(1,n_rows,n_cols,1)
        
        while not done and ep_step < agent.ep_trial_step:
            
            ep_step += 1
            agent.step += 1
            
            action_arr, action = agent.get_action(state)
            
            next_state, reward, done = game.frame_step(action, ep_step)
            
            rewards += reward
            
            next_state = next_state.reshape(1,n_rows,n_cols,1)
            
            state = next_state
            
            agent.score = rewards
                    
            if done:
                agent.episode += 1
                scores.append(agent.score)
                episodes.append(agent.episode)
                avg_score = np.mean(scores[-min(30, len(scores)):])
                print('episode :{:>6,d}'.format(agent.episode),'/ ep step :{:>5,d}'.format(ep_step), \
                      '/ rewards :{:>4.1f}'.format(rewards),'/ last 30 avg :{:> 4.1f}'.format(avg_score) )
                break
                
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
