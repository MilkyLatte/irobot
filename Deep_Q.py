import torch
from torch import nn
import torch.nn.functional as F


import gym
import numpy as np
import math
import cv2
from collections import deque, namedtuple
import random

from wrappers import wrap_deepmind


env = gym.make('PongDeterministic-v4')
# env = gym.make('BeamRider-v0')
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

n_actions = env.action_space.n

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
        self.out = nn.Linear(512, n_actions)


    
    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.out(x)
        return x


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
        self.env = wrap_deepmind(env)
        
        self.agent_history_length = agent_history_length
        self.state = None
        self.last_lives = 0

    def reset(self):
        frame = self.env.reset()
        frame = frame.reshape(1, HEIGHT, WIDTH)
                
        self.state = np.repeat(frame, self.agent_history_length, axis=0)
        
        return torch.tensor(self.state.reshape(-1, 4, HEIGHT, WIDTH), device=device), frame
    
    def step(self, action):
        new_frame, reward, terminal, info = self.env.step(action) 
        new_frame = new_frame.reshape(1, HEIGHT, WIDTH)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']
        new_state = np.append(
            self.state[1:, :, :], new_frame, axis=0)
        self.state = new_state
       
        return torch.tensor(self.state.reshape(-1, 4, HEIGHT, WIDTH), device=device), new_frame, reward, terminal, terminal_life_lost, info


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
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
NUMBER_OF_FRAMES = 15000000
TARGET_UPDATE = 10000
ANNELING_FRAMES = 1000000

HEIGHT = 84
WIDTH = 84
EPSILON = 0
MEM_SIZE = 1000000
LEARNING_STARTS = 50000

UPDATE_FREQ = 4
LEARNING_RATE = 0.00025


policy_net = DeepQNet(HEIGHT, WIDTH).to(device)
target_net = DeepQNet(HEIGHT, WIDTH).to(device)

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEM_SIZE)
steps_done = 0

def get_epsilon(current_step):
    rate = (EPS_END-EPS_START)/ANNELING_FRAMES
    eps_threshold = rate * current_step + EPS_START
    if eps_threshold < EPS_END:
        return EPS_END
    return eps_threshold

def select_action(state, steps_done, eval=False):
    global EPSILON

    # This equation is for the decaying epsilon
    eps_threshold = 0
    if eval:
        eps_threshold = EPS_END
    r = np.random.rand()

    if steps_done <= LEARNING_STARTS:
        eps_threshold = EPS_START
    else:
        eps_threshold = get_epsilon(steps_done - LEARNING_STARTS)
    EPSILON = eps_threshold
    # We select an action with an espilon greedy policy 
    if r > eps_threshold:
        with torch.no_grad():
            # Return the action with the maximum Q value for the current state
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device)


def optimize_model():
    if memory.current < LEARNING_STARTS:
        return 0
    optimizer.zero_grad()

    states, actions, rewards, next_states, dones = memory.get_minibatch()
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done = torch.tensor(dones).to(device)

    state_action_values = policy_net(states_v).gather(
        1, actions_v.long().unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]
    next_state_values[done] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
     
    # Compute Huber loss
    # Optimize the model
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    loss.backward()
    optimizer.step() 
    return loss.item()

PATH = "./deepQ.pt"

def train_model(num_frames):
    env = Atari('PongDeterministic-v4')
    
    cumulative_frames = 0
    best_score = 0
    games = 0
    full_loss = []
    rewards = []
    while 1:
        state, memory_state = env.reset()
        done = False
        cum_reward = 0
        cum_loss = []
        while not done:
            action = select_action(state, cumulative_frames)

            next_state, memory_next_state, reward, done, life_lost, _ = env.step(action.item())

            memory.add_experience(action.item(), memory_state.reshape(HEIGHT, WIDTH), reward, life_lost)


            state = next_state
            memory_state = memory_next_state
            if cumulative_frames % UPDATE_FREQ == 0 and cumulative_frames > LEARNING_STARTS:
                loss = optimize_model()
                cum_loss.append(loss)
            cum_reward += reward
            cumulative_frames += 1
        
        if best_score < cum_reward:
            best_score = cum_reward
        if len(cum_loss) == 0:
            full_loss.append(0)
        else:
            full_loss.append(np.mean(cum_loss))
        rewards.append(cum_reward)
        games += 1
        if cumulative_frames % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if games % 10 == 0:
            print("=============================================")
            print("Game: {} | Frame {}".format(games, cumulative_frames))
            print("Final reward: {}".format(cum_reward))
            print("Epsilon after: {}".format(EPSILON))
            print("Best High Score: {}".format(best_score))
            print("Avg Loss Last 100 games: {}".format(np.mean(full_loss[-100:])))
            print("Avg Reward Last 100 games: {}".format(np.mean(rewards[-100:])))
        
        if np.mean(rewards[-100:]) >= 18 and cumulative_frames > LEARNING_STARTS:
            break

    torch.save(target_net.state_dict(), PATH)

def load_agent():
    model = DeepQNet(HEIGHT, WIDTH).to(device)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    return model


def inference(episodes, model):
    env = Atari('BreakoutNoFrameskip-v4')
    for episode in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            env.env.render()
            with torch.no_grad():
                action = select_action(observation, True)
                observation, _, reward, done, _, _ = env.step(action.item())
                if reward != 0:
                    print(reward)

def main():
    train_model(NUMBER_OF_FRAMES)
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
