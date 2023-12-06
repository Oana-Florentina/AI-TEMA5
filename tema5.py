import numpy as np
import random

class Action():
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3

    COUNT = 4
    arrows = ['←', '→', '↓', '↑']

class State():
    lines = 7
    columns = 10
    finish_pos = (3, 7)
    start_pos = (3, 0)
    finish_reward = 20
    default_reward = -1
    actions = {
        Action.LEFT: (0, -1),
        Action.RIGHT: (0, 1),
        Action.DOWN: (1, 0),
        Action.UP: (-1, 0)
    }

    def __init__(self, agent_pos, wind):
       self.agent_pos = agent_pos
       self.wind = wind 

    def __int__(self):
        return int(self.agent_pos[0] * self.columns + self.agent_pos[1])
    
    def apply_wind(state_pos, wind_val):
        next_line = max(0, state_pos[0] - wind_val)
        next_col = state_pos[1]

        return (next_line, next_col)

    def get_next_state(self, action):
        if self.is_valid_action(action):
            next_pos = tuple(np.add(self.agent_pos, self.actions[action]))
            next_pos = State.apply_wind(next_pos, self.wind[self.agent_pos[1]])
            return State(next_pos, self.wind)
        
        return self 
    
    def is_final_state(self):
        if self.agent_pos==self.finish_pos:
            return True
        return False
    
    def get_reward(self, action):
        next_state = self.get_next_state(action)
        if next_state.is_final_state():
            return self.finish_reward
        return self.default_reward

    def is_valid_action(self, action):
        next_pos =  tuple(np.add(self.agent_pos, self.actions[action]))
        if next_pos[0]>=self.lines or next_pos[0]<0:
            return False
        if next_pos[1]>=self.columns or next_pos[1]<0:
            return False
        return True
      

class QLearn():
    def __init__(self, start_state, alpha=0.7, gamma=0.95, episode_count=10000, min_epsilon=0.1, decay_rate=0.005):
        self.start_state = start_state 
        self.alpha = alpha 
        self.gamma = gamma 
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.episode_count = episode_count
        
        self.epsilon = 1.0
        
        self.q_table = np.zeros((State.lines * State.columns, Action.COUNT))

    def get_epsilon_greedy_action(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            rand_action = random.randint(0, Action.COUNT - 1)  
            return rand_action
        return np.argmax(self.q_table[int(state)])

    def decrease_epsilon(self):
        if self.epsilon - self.decay_rate >= self.min_epsilon:
            self.epsilon -= self.decay_rate

    def train(self):
        for episode in range(self.episode_count):
            curr_state = self.start_state          
            while True:
                if curr_state.is_final_state():
                    break
                
                action = self.get_epsilon_greedy_action(curr_state)
                next_state = curr_state.get_next_state(action)
                
                self.q_table[int(curr_state)][action] = (1 - self.alpha) * self.q_table[int(curr_state)][action] + self.alpha * (curr_state.get_reward(action) + self.gamma * max(self.q_table[int(next_state)])) 

                curr_state = next_state
            self.decrease_epsilon()

    def print_policy(self):
        for i in range(State.lines):
            for j in range(State.columns):
                action = np.argmax(self.q_table[i * State.columns + j])
                print(Action.arrows[action], end=' ')
            print()
            
wind_effect = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

start_state = State((3, 0), wind_effect)
model = QLearn(start_state)
model.train()
model.print_policy()