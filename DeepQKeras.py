from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf

import gym
import numpy as np
import math
import cv2
import random





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

        return self.states.reshape(-1, self.frame_height, self.frame_width, self.agent_history_length), self.actions[self.indices], self.rewards[self.indices], self.new_states.reshape(-1, self.frame_height, self.frame_width, self.agent_history_length), self.terminal_flags[self.indices]


class Atari(object):
    def __init__(self, envName, agent_history_length=4):
        self.env = gym.make(envName)
        self.agent_history_length = agent_history_length
        self.state = None
        self.last_lives = 0

    def reset(self):
        frame = self.env.reset()

        processed_frame = convert_screen(frame)
        self.state = np.repeat(
            processed_frame, self.agent_history_length, axis=0)

        return self.state.reshape(-1, HEIGHT, WIDTH, 4), processed_frame

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

        return self.state.reshape(-1, HEIGHT, WIDTH, 4), processed_new_frame, reward, terminal, terminal_life_lost, info


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
EPS_END = 0.05
NUMBER_OF_FRAMES = 5000000
TARGET_UPDATE = 10000


HEIGHT = 84
WIDTH = 84
EPSILON = 0
MEM_SIZE = np.floor(NUMBER_OF_FRAMES * 0.1)

policy_net = DeepQNet(n_actions, HEIGHT, WIDTH)
target_net = DeepQNet(n_actions, HEIGHT, WIDTH)

memory = ReplayMemory(int(MEM_SIZE))
steps_done = 0


def get_epsilon(current_step):
    rate = (EPS_END-EPS_START)/MEM_SIZE
    eps_threshold = rate * current_step + EPS_START
    if eps_threshold < EPS_END:
        return EPS_END
    return eps_threshold


def select_action(state, eval=False):
    global steps_done
    global EPSILON

    # This equation is for the decaying epsilon
    eps_threshold = get_epsilon(steps_done)
    steps_done += 1

    if eval:
        eps_threshold = EPS_END

    r = np.random.rand()

    EPSILON = eps_threshold

    # We select an action with an espilon greedy policy
    if r > eps_threshold:
        # Return the action with the maximum Q value for the current state
        return np.argmax(policy_net.model.predict(state)[0])
    else:
        return random.randrange(n_actions)


def optimize_model():
    if memory.current < memory.batch_size:
        return 0
    states, actions, rewards, next_states, dones = memory.get_minibatch()
    
    state_action_values = policy_net.model.predict(states)
    next_state_values = np.amax(target_net.model.predict(next_states), axis=1)
    next_state_values[dones] = 0.0
    
    expected_state_action_values = next_state_values * GAMMA + rewards
    state_action_values[:,actions] = expected_state_action_values
    # Compute Huber loss
    history = policy_net.model.fit(states, state_action_values, verbose=False)

    return history.history['loss'][0]



PATH = "./deepQ.pt"


def train_model(num_frames):
    env = Atari('BreakoutNoFrameskip-v4')
    cumulative_frames = 0
    best_score = 0
    game_scores = []
    losses = []
    games = 0
    while cumulative_frames < num_frames:
        print("=============================================")
        games += 1
        print("GAME: {}".format(games))
        state, memory_state = env.reset()
        done = False
        game_score = 0
        current_frames = cumulative_frames
        cum_reward = 0
        cum_loss = 0

        while not done:
            action = select_action(state)

            next_state, memory_next_state, reward, done, life_lost, _ = env.step(
                action)
            game_score += reward

            reward = clip_reward(reward)

            if done:
                next_state = None

            memory.add_experience(action, memory_state.reshape(
                HEIGHT, WIDTH), reward, life_lost)

            state = next_state
            memory_state = memory_next_state
            loss = optimize_model()
            cum_reward += reward
            cumulative_frames += 1
            cum_loss += loss

        game_scores.append(game_score)
        losses.append(cum_loss/(cumulative_frames-current_frames))
        if best_score < game_score:
            print("NEW HIGHSCORE: {}".format(game_score))
            best_score = game_score
        print("Current Frame: {}".format(cumulative_frames))
        print("Avg Episode Loss:{}".format(
            cum_loss/(cumulative_frames-current_frames)))
        print("Final reward: {}".format(cum_reward))
        print("Epsilon after: {}".format(EPSILON))
        print("Final Game Score: {}".format(game_score))
        print("Best High Score: {}".format(best_score))
        try:
            print("Average game score last 15 games: {}".format(np.mean(game_scores[-15:])))
            print("Average loss 15 games {}".format(np.mean(losses[-15:])))
        except:
            pass
        if cumulative_frames % TARGET_UPDATE == 0:
            target_net.model.set_weights(policy_net.model.get_weights())

    target_net.model.save('my_model.h5')


# def load_agent():
#     model = DeepQNet(HEIGHT, WIDTH).to(device)
#     model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
#     model.eval()
#     return model


# def inference(episodes, model):
#     env = Atari('BreakoutNoFrameskip-v4')
#     for episode in range(episodes):
#         observation, _ = env.reset()
#         done = False
#         while not done:
#             env.env.render()
#             with torch.no_grad():
#                 action = select_action(observation, True)
#                 observation, _, reward, done, _, _ = env.step(action.item())
#                 if reward != 0:
#                     print(reward)


def main():
    train_model(NUMBER_OF_FRAMES)
    # model = load_agent()
    # inference(100, model)

if __name__ == '__main__':
    main()

