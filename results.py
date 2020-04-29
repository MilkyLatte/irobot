from datetime import datetime
import os
import torch
import pickle
import zlib
import gym
import numpy as np
import imageio

class results():
    def __init__(self, global_vars_dict : dict, folder):
        super().__init__()
        self.time_start = datetime.now()
        # create the results directory
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.log_file_name = os.path.join(folder, 'log.txt')

        # write a brief descriptor 
        with open(self.log_file_name, mode='w') as f:
            f.write('Train started: ' + self.time_start.strftime('%Y%m%d_%H%M') + '\n')
            f.writelines(f'{k} = {v}\n' for k, v in global_vars_dict.items() if k==k.upper())
            f.write('\ngames,frames,epsilon,score,network_loss,delta_t,frames/t\n')

        self.last_t = self.time_start
        self.last_frames = 0
        self.records = []
        
    def record(self, frames, games, epsilon, score, network_loss):
        # can't derail training for some silly bug so wrap with try...
        try:            
            t = datetime.now()
            delta_t = (t - self.last_t).total_seconds()
            delta_frames = (frames - self.last_frames)
            frames_per_sec = delta_frames / max(delta_t, 1e-6)
            self.last_t = t
            self.last_frames = frames
            # self.best_score = max(score, self.best_score)
            with open(self.log_file_name, mode='a') as f:
                f.write(t.strftime('%H:%M:%S') + f',{games},{frames},{epsilon},{score},{network_loss},{delta_t},{frames_per_sec}\n')
            self.records.append((t, games, frames, epsilon, score, network_loss, delta_t, frames_per_sec))
            if len(self.records)%10==0:
                self.save_results()
                print(f'episodes {len(self.records)} frames/sec {frames_per_sec} last score {score}')
        except Exception as err:
            print(f'Error recording results {err}')

    def save_model(self, episode_idx, q_network):
        # periodic save of model into same directory
        try:
            file_name = os.path.join(self.folder, f'torch_model_episode_{episode_idx}.pyt')
            torch.save(q_network.state_dict(), file_name)

            # play a single game and save a gif too
        except Exception as err:
            print(f'Error saving model {err}')

    def save_single_game(self, episode_idx, data):
        # data = tuples of (frame_idx, max_q, action, reward, rgb_frame)
        file_name = os.path.join(self.folder, f'single_game_data_{episode_idx}.zpkl')
        tmp = pickle.dumps(data)
        ztmp = zlib.compress(tmp)
        with open(file_name, 'wb') as f:
            f.write(ztmp)
        # generate a gif and save that too:
        file_name = os.path.join(self.folder, f'gif_video_{episode_idx}.gif')
        imageio.mimsave(file_name, [x[4] for x in data], fps=60)

    def save_results(self):
        file_name = os.path.join(self.folder, 'records.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(self.records, f)

    def close(self):
        self.save_results()


def test_results():
    r = results(globals(), 'logs')
    r.record(100, 10, 0.9, 3, 0.32)
    r.record(101, 11, 0.89, 4, 0.22)
    r.record(102, 12, 0.88, 6, 0.21)
    r.record(104, 13, 0.87, 6, 0.18)
    r.close()

if __name__ =='__main__':
    test_results()
