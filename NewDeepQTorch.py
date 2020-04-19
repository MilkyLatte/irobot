import torch
from torch import nn
import torch.nn.functional as F

import gym
import numpy as np
import math
import cv2
from collections import deque, namedtuple
import random
import time
import results

#for GIF generation
import imageio
from skimage.transform import resize

from wrappers import wrap_deepmind, make_atari


e = gym.make('PongNoFrameskip-v4')
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

n_actions = e.action_space.n
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EXP_FRACTION = 0.1
NUMBER_OF_FRAMES = 1e7

TARGET_UPDATE = 1000
# ANNELING_FRAMES = 1000000

TRAIN_FREQUENCY = 4

HEIGHT = 84
WIDTH = 84
EPSILON = 0
MEM_SIZE = 10000
LEARNING_STARTS = 10000
SINGLE_EVAL_STEPS = 10_000

SCHEDULE_TIMESTEPS = EXP_FRACTION * NUMBER_OF_FRAMES

LEARNING_RATE = 1e-4
PATH = "./deepQ.pt"
GIF_PATH = './game_video/'

class DeepQNet(nn.Module):
    def __init__(self, h, w):
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


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes)


policy_net = DeepQNet(HEIGHT, WIDTH).to(device)
target_net = DeepQNet(HEIGHT, WIDTH).to(device)

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEM_SIZE)
steps_done = 0


def get_epsilon(current_step):
    fraction = min(float(current_step) / SCHEDULE_TIMESTEPS , 1.0)
    eps = EPS_START + fraction * (EPS_END - EPS_START)
    if eps < EPS_END:
        eps = EPS_END
    return eps
    # rate = (EPS_END-EPS_START)/ANNELING_FRAMES
    # eps_threshold = rate * current_step + EPS_START
    # if eps_threshold < EPS_END:
    #     return EPS_END
    # return eps_threshold


def select_action(state, steps_done):
    global EPSILON
    # This equation is for the decaying epsilon
    eps_threshold = 0
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
            return policy_net(state).max(1)[1].item()
    else:
        return np.random.randint(0, n_actions)


def optimize_model(t):
    if t < LEARNING_STARTS:
        return 0

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)


    states = np.array(states).reshape(-1, 4, HEIGHT, WIDTH)
    next_states = np.array(next_states).reshape(-1, 4, HEIGHT, WIDTH)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.BoolTensor(dones).to(device)

    curr_Q = policy_net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
    max_next_Q = target_net(next_states).max(1)[0]
    max_next_Q[dones] = 0.0
    max_next_Q = max_next_Q.detach()
    expected_Q = rewards + GAMMA * max_next_Q

    # Compute Huber loss
    loss = F.smooth_l1_loss(curr_Q,
                            expected_Q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


def train_model(num_frames):
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env,episode_life=True, frame_stack=True)
    train_results = results.results(globals())

    cumulative_frames = 0
    best_score = -50
    games = 0
    full_loss = []
    rewards = []
    while 1:
        state = env.reset()
        done = False
        cum_reward = 0
        cum_loss = []
        while not done:
            action = select_action(torch.tensor(np.array(state).reshape(-1, 4, HEIGHT, WIDTH)).to(device), cumulative_frames)

            next_state, reward, done, _ = env.step(action)

            memory.add(state, action, reward, next_state, reward)

            state = next_state
            if cumulative_frames % TRAIN_FREQUENCY == 0 and cumulative_frames > LEARNING_STARTS:
                loss = optimize_model(cumulative_frames)
                cum_loss.append(loss)
                        
            cum_reward += reward
            cumulative_frames += 1
        
            if cumulative_frames % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if best_score < cum_reward:
            best_score = cum_reward
        if len(cum_loss) == 0:
            full_loss.append(0)
        else:
            full_loss.append(np.mean(cum_loss))
        rewards.append(cum_reward)
        games += 1
        # Single Game Evaluation for GIF 
        if games % 1000 == 0:    
            print("Evaluating for gif")    
            terminal = False
            real_frames_for_gif = []
            frames_for_gif = []
            eval_rewards = []
            frame = env.reset()

            #playing one game
            while not terminal:
                episode_reward_sum = 0
                single_action = select_action(torch.tensor(np.array(frame).reshape(-1, 4, HEIGHT, WIDTH)).to(device), cumulative_frames)
                new_frame, reward, terminal, _ = env.step(single_action)

                real_frame = env.render(mode='rgb_array')                
                real_frames_for_gif.append(real_frame)
                frames_for_gif.append(new_frame)
                frame = new_frame
                episode_reward_sum += reward
                if terminal:
                    eval_rewards.append(episode_reward_sum)    
            try:
                    generate_gif(cumulative_frames, real_frames_for_gif, eval_rewards[0], PATH)
                    generate_gif(cumulative_frames+1, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                    print("No evaluation game finished")

        # Printing Game Progress
        if games % 10 == 0:
            print("=============================================")
            print("Game: {} | Frame {}".format(games, cumulative_frames))
            print("Final reward: {}".format(cum_reward))
            print("Epsilon after: {}".format(EPSILON))
            print("Best High Score: {}".format(best_score))
            print("Avg Loss Last 100 games: {}".format(
                np.mean(full_loss[-100:])))
            print("Avg Reward Last 100 games: {}".format(
                np.mean(rewards[-100:])))

        train_results.record(cumulative_frames, games, EPSILON, cum_reward, full_loss[-1])

        if np.mean(rewards[-100:]) >= 18 and cumulative_frames > LEARNING_STARTS:
            break

    # torch.save(target_net.state_dict(), PATH)
    train_results.close()


def load_agent():
    model = DeepQNet(HEIGHT, WIDTH).to(device)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def eval_action(state, model):
    r = np.random.rand()
    if r > EPS_END:
        with torch.no_grad():
            # Return the action with the maximum Q value for the current state
            return model(state).max(1)[1].item()
    else:
        return np.random.randint(0, n_actions)

def inference(episodes, model, env_name):
    env = make_atari(env_name)
    env = wrap_deepmind(env, episode_life=True, frame_stack=True)  
    for _ in range(episodes):
        observation = env.reset()
        done = False
        while not done:
            time.sleep(0.05)
            env.render()
            observation = torch.tensor(np.array(observation).reshape(-1, 4, HEIGHT, WIDTH)).to(device)
            with torch.no_grad():
                action = model(observation).max(1)[1].item()
                observation, reward, done, _ = env.step(action)
                if reward != 0:
                    print(reward)


def generate_gif(current_frame, frames_for_gif, reward, path):
    #takes current, and generates and saves a GIF to PATH for input frames
    #for i, frame_i in enumerate(frames_for_gif): 
    #    frame = np.array(frame_i)[None]
    #    frames_for_gif[i] = resize(frame, (420, 320,3), 
    #                                 ).astype(np.uint8)      
    imageio.mimsave(f'{GIF_PATH}{"ATARI_frame_{0}_reward_{1}.gif".format(current_frame, reward)}',
                        frames_for_gif, duration=1/30)

def main():
    train_model(NUMBER_OF_FRAMES)
    #model = load_agent()
    #inference(1, model, 'PongNoFrameskip-v4')

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
