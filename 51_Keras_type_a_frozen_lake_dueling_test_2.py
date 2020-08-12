import sys
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque
from keras.layers import *
from keras.models import Sequential,Model
import keras
from keras import backend as K_back
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D

state_size = 64
action_size = 5

model_path = "save_model/"
graph_path = "save_graph/"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
load_model = True

class DQN_agent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # train time define
        self.training_time = 30*60
        
        self.episode = 0
        
        # These are hyper parameters for the DQN_agent
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.049
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 500
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        self.input_shape = (9,9,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

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

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate:
            # print("Random action selected!!")
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.zeros((self.batch_size, 9, 9, 1))
        next_states = np.zeros((self.batch_size, 9, 9, 1))
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
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(tgt_q_value_next[i]))
                
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
                        
def main():
    
    # DQN_agent 에이전트의 생성
    agent = DQN_agent(state_size, action_size)
    if load_model:
        agent.model.load_weights(model_path + "/Model_dueling_0.h5")
    
    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agent.ep_trial_step)
    last_n_game_reward = deque(maxlen=20)
    avg_ep_step = np.mean(last_n_game_score)
    avg_reward = 0
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agent.episode = 0
    time_step = 0
    
    while time.time() - start_time < agent.training_time and avg_reward < 450:
        
        done = False
        ep_step = 0
        rewards = 0
                
        state = np.zeros((9,9))
        
        rand_init = np.random.randint(low=0, high=3)
        # rand_init = 0
        state[2][(rand_init+1)%9] = 2
        state[4][(rand_init+4)%9] = 2
        state[6][(rand_init+7)%9] = 2
        state[8][8] = 4
        state[0][0] = 5
        
        # print(state)
        
        agent_row = 0
        agent_col = 0
        
        state = state.reshape(1,9,9,1)
        
        while not done and ep_step < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"
            else :
                agent.progress = "Training"

            ep_step += 1
            time_step += 1
            
            action = agent.get_action(state)
            
            # print("action :", action)
            if action == 0:
                if (agent_row + 1) < 9:
                    agent_row += 1
            if action == 1:
                if agent_row > 0:
                    agent_row -= 1
            if action == 2:
                if agent_col > 0:
                    agent_col -= 1
            if action == 3:
                if (agent_col+1) < 9:
                    agent_col += 1
            
            agent_pos = np.zeros((9,9))
            agent_pos[agent_row][agent_col] = 5
            
            ice_lake = np.zeros((9,9))
            hole_1_col = int((rand_init+ep_step+1)%9)
            hole_2_col = int((rand_init+ep_step+4)%9)
            hole_3_col = int((rand_init+ep_step+7)%9)
            ice_lake[2][hole_1_col] = 2
            ice_lake[4][hole_2_col] = 2
            ice_lake[6][hole_3_col] = 2
            ice_lake[8][8] = 4
            
            next_state = agent_pos + ice_lake
            # print(next_state)
            
            next_state_t = next_state.reshape(1,9,9,1)
            state = next_state_t
            
            # reward = agent_row - 8 + agent_col - 8
            reward = -1
            
            if np.count_nonzero(next_state == 7) > 0:
                if ep_step < 15:
                    reward = reward - 200
                else:
                    reward = reward - 100
                # done = True
                
            if np.count_nonzero(next_state == 9) > 0:
                done = True
                reward = 500
            
            rewards += reward
            
            agent.append_sample(state, action, reward, next_state_t, done)
                        
            if agent.progress == "Training":
                agent.train_model()
                if done or ep_step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()
                    
            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    # print(next_state)
                    agent.episode += 1
                    last_n_game_score.append(ep_step)
                    last_n_game_reward.append(rewards)
                    avg_ep_step = np.mean(last_n_game_score)
                    avg_reward = np.mean(last_n_game_reward)
                print("episode :{:>5d} / ep_step :{:>5d} / reward :{:>4.1f} / avg. rewards :{:>4.0f}".format(agent.episode, ep_step, rewards, avg_reward))
                break
                
    agent.model.save_weights(model_path + "/Model_dueling_0.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
