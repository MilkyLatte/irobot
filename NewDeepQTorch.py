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

env = gym.make('BreakoutNoFrameskip-v4')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = env.action_space.n

Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'done', 'new_state'])

class DeepQNet(nn.Module):
    def __init__(self, n_actions, h, w):
        super(DeepQNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(
            conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
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



class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)




BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
TARGET_UPDATE = 10000


HEIGHT = 84
WIDTH = 84
EPSILON = 0
MEM_SIZE = 1000000
LEARNING_STARTS = 50000

LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

policy_net = DeepQNet(n_actions, HEIGHT, WIDTH).to(device)
target_net = DeepQNet(n_actions, HEIGHT, WIDTH).to(device)


optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
memory = ReplayMemory(int(MEM_SIZE))


def get_epsilon(current_step):
    rate = (EPS_END-EPS_START)/MEM_SIZE
    eps_threshold = rate * current_step + EPS_START
    if eps_threshold < EPS_END:
        return EPS_END
    return eps_threshold


def select_action(state, steps_done, eval=False):
    global EPSILON

    # This equation is for the decaying epsilon
    eps_threshold = get_epsilon(steps_done)

    if eval:
        eps_threshold = EPS_END

    r = np.random.rand()

    EPSILON = eps_threshold

    # We select an action with an espilon greedy policy
    if r > eps_threshold:
        with torch.no_grad():
            # Return the action with the maximum Q value for the current state
            return policy_net(torch.tensor(state).to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device)


def optimize_model():
    if memory.__len__() < LEARNING_STARTS:
        return 0

    states, actions, rewards, dones, next_states = memory.sample(BATCH_SIZE) 
    states_v = torch.tensor(states.reshape(-1, 4, 84, 84)).to(device)
    next_states_v = torch.tensor(next_states.reshape(-1, 4, 84, 84)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done = torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = policy_net(states_v).gather(
        1, actions_v.long().unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]
    next_state_values[done] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(num_frames):
    env = gym.make('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env)
    cumulative_frames = 0

    highest_score = 0
    current_cum_loss = []
    current_game_score = 0
    new_game = True
    games = 0

    losses = []
    scores = []
    while cumulative_frames < num_frames:
        if new_game:
            print("============================")
            print("Game: {} | Frame {}".format(games, cumulative_frames))
            new_game = False
        state = env.reset()
        done = False

        while not done:
            action = select_action(
                state.__array__().reshape(-1, 4, 84, 84), cumulative_frames).item()

            next_state, reward, done, info = env.step(action)

            memory.append(Experience(state, action, reward, done, next_state))

            state = next_state
            loss = optimize_model()

            current_cum_loss.append(loss)

            current_game_score += reward
            cumulative_frames += 1

            if info['ale.lives'] == 0:
                if highest_score < current_game_score:
                    highest_score = current_game_score

                current_loss = np.mean(current_cum_loss)
                losses.append(current_loss)
                scores.append(current_game_score)

                print("Current game score: {}".format(current_game_score))
                print("Current loss: {}".format(current_loss))
                print("Highest Score: {}".format(highest_score))
                print("Average loss last 50 games: {}".format(
                    np.mean(losses[-50:])))
                print("Average score last 50 games: {}".format(
                    np.mean(scores[-50:])))

                current_game_score = 0
                current_cum_loss = []
                new_game = True
                games += 1
        if cumulative_frames % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(target_net.state_dict(), './net.pt')
    return losses, scores


def main():
    train_model(1000000)
if __name__ == '__main__':
    main()
