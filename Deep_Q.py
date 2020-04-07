import torch
from torch import nn
import torch.nn.functional as F

import gym
import numpy as np
import math
import cv2
from collections import deque, namedtuple
import random



env = gym.make('BeamRider-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ACTION_MEANING = {
#     0: "NOOP",
#     1: "FIRE",
#     2: "UP",
#     3: "RIGHT",
#     4: "LEFT",
#     5: "DOWN",
#     6: "UPRIGHT",
#     7: "UPLEFT",
#     8: "DOWNRIGHT",
#     9: "DOWNLEFT",
#     10: "UPFIRE",
#     11: "RIGHTFIRE",
#     12: "LEFTFIRE",
#     13: "DOWNFIRE",
#     14: "UPRIGHTFIRE",
#     15: "UPLEFTFIRE",
#     16: "DOWNRIGHTFIRE",
#     17: "DOWNLEFTFIRE",
# }

actions = [0, 1, 3, 4]
n_actions = 4
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))





class DeepQNet(nn.Module):
    def __init__(self, h, w):
        super(DeepQNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size= 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(h, 8, 4), 4, 2), 3, 1)

        linear_input = convh * convw * 64
        self.fc1 = nn.Linear(linear_input, 512)
        self.out = nn.Linear(512, 4)


    
    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.out(x)
        return x


def clip_reward(reward):
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1


class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=1000000, frame_height=84, frame_width=84,
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty(
            (self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.frames[index-self.agent_history_length+1:index+1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(
                    self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return self.states, self.actions[self.indices], self.rewards[self.indices], self.new_states, self.terminal_flags[self.indices]



class Atari(object):
    def __init__(self, envName, agent_history_length=4):
        self.env = gym.make(envName)
        self.agent_history_length = agent_history_length
        self.state = None
        self.last_lives = 0

    def reset(self):
        frame = self.env.reset()
        
        processed_frame = convert_screen(frame)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=0)
        
        return torch.tensor(self.state.reshape(-1, 4, HEIGHT, WIDTH), device=device), processed_frame
    
    def step(self, action):
        new_frame, reward, terminal, info = self.env.step(action) 
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        processed_new_frame = convert_screen(new_frame)
        new_state = np.append(
            self.state[1:, :, :], processed_new_frame, axis=0)
        self.state = new_state
       
        return torch.tensor(self.state.reshape(-1, 4, HEIGHT, WIDTH), device=device), processed_new_frame, reward, terminal, terminal_life_lost, info


def convert_screen(screen):
    # This function simplifies the environment as color is not important
    # the top sides of the screen are also irrelevant as the agent will
    # get a reward directly from the environment and not by looking at
    # the score
    reshaped = cv2.resize(screen, (84, 110), interpolation=cv2.INTER_AREA)
    cropped = reshaped[20:104]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # cv2.imshow("cropped", gray)
    # cv2.waitKey(0)
    # We reshape as pytorch uses the order of features (CHW)
    return gray.reshape(1, HEIGHT, WIDTH)


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.1
DECAY_RATE = -0.0000009
TARGET_UPDATE = 10000

HEIGHT = 84
WIDTH = 84
EPSILON = 0

policy_net = DeepQNet(HEIGHT, WIDTH).to(device)
target_net = DeepQNet(HEIGHT, WIDTH).to(device)

optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=0.00001)
memory = ReplayMemory(1000000)
steps_done = 0

def get_epsilon(rate, current_step):
    eps_threshold = rate * current_step + 1
    if eps_threshold < EPS_END:
        return EPS_END
    return eps_threshold

def select_action(state):
    global steps_done
    global EPSILON

    # This equation is for the decaying epsilon
    eps_threshold = get_epsilon(DECAY_RATE, steps_done)
    steps_done += 1

    r = np.random.rand()

    EPSILON = eps_threshold

    # We select an action with an espilon greedy policy 
    if r > eps_threshold:
        with torch.no_grad():
            # Return the action with the maximum Q value for the current state
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device)


def optimize_model():
    if memory.current < memory.batch_size:
        return 0
    states, actions, rewards, next_states, terminals = memory.get_minibatch()
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)

    non_final_mask = terminals != None
    non_final_next_states = torch.from_numpy(next_states[non_final_mask]).to(device=device)
                 
    state_batch = torch.from_numpy(states).to(device=device)
    action_batch = torch.from_numpy(actions).to(device=device).reshape((memory.batch_size, 1))
    reward_batch = torch.from_numpy(rewards).to(device=device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.long())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.item()

PATH = "./deepQ.pt"

def train_model(num_frames):
    env = Atari('BeamRiderDeterministic-v4')
    cumulative_frames = 0
    while cumulative_frames < num_frames:
        print("=============================================")
        state, memory_state = env.reset()
        done = False
        game_score = 0
        current_frames = cumulative_frames
        cum_reward = 0
        cum_loss = 0
        while not done:
            action = select_action(state)

            next_state, memory_next_state, reward, done, life_lost, _ = env.step(actions[action.item()])
            game_score += reward
            if life_lost:
                reward = -1
            reward = clip_reward(reward)
           
            if done:
                next_state = None

            memory.add_experience(action.item(), memory_state.reshape(HEIGHT, WIDTH), reward, life_lost)


            state = next_state
            memory_state = memory_next_state
            loss = optimize_model()
            cum_reward += reward
            cumulative_frames += 1
            cum_loss += loss
        print("Current Frame: {}".format(cumulative_frames))
        print("Avg Episode Loss:{}".format(cum_loss/(cumulative_frames-current_frames)))
        print("Final reward: {}".format(cum_reward))
        print("Epsilon after: {}".format(EPSILON))
        print("Cumulative Frames: {}".format(cumulative_frames))
        print("Final Game Score: {}".format(game_score))
        if cumulative_frames % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(target_net.state_dict(), PATH)

def load_agent():
    model = DeepQNet(HEIGHT, WIDTH).to(device)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


def inference(episodes, model):
    for episode in range(episodes):
        observation = env.reset()
        done = False
        while not done:
            env.render()
            with torch.no_grad():
                state = torch.tensor(convert_screen(
                    observation).reshape(-1, 1, 84, 84), device=device)
                r = np.random.rand()
                if r < 0.9:
                    action = actions[model(state).max(1)[1].view(1, 1).item()]
                else:
                    action = np.random.choice(actions)
                observation, _, done, _ = env.step(action)

import time
def main():
    train_model(10000000)
    # model = load_agent()
    # inference(100, model)



        

if __name__ == '__main__':
    main()
        
    





# for i_episode in range(100):
#     observation = env.reset()
#     done = False
#     print("HERE")

#     while not done:
#         env.render()    
#         action = env.action_space.sample()
#         # action = np.random.randint(3,5)
#         observation, reward, done, info = env.step(action)
#         if reward > 0:
#             print(reward)
#         elif reward < 0:
#             print(reward)
#         if done:
#             print("Episode finished")
# env.close()
