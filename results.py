from datetime import datetime
import os
import torch
import pickle
# import zlib

class results():
    def __init__(self, global_vars_dict : dict, folder='logs'):
        super().__init__()
        # create a log file in the log directory
        self.time_start = datetime.now()
        self.log_file_name = 'results_' + self.time_start.strftime('%Y%m%d_%H%M') + ".log"
        self.log_file_path = os.path.join(folder, self.log_file_name)
        if not os.path.exists(os.path.dirname(self.log_file_path)):
            os.makedirs(os.path.dirname(self.log_file_path))

        # write a brief descriptor 
        with open(self.log_file_path, mode='w') as f:
            f.write('Train started: ' + self.time_start.strftime('%H:%M') + '\n')
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
            with open(self.log_file_path, mode='a') as f:
                f.write(t.strftime('%H:%M:%S') + f',{games},{frames},{epsilon},{score},{network_loss},{delta_t},{frames_per_sec}\n')
            self.records.append((t, games, frames, epsilon, score, network_loss, delta_t, frames_per_sec))
            if (len(self.records)%5==0):
                self.save_results()
            print(f'record count {len(self.records)} frames/sec {frames_per_sec}')
        except Exception as err:
            print(f'Error recording results {err}')

    def save_model(self, frame_idx, q_network):
        # periodic save of model into same directory
        try:
            file_name = self.log_file_path.replace('.log', f'_torch_{frame_idx}.pyt')
            torch.save(q_network.state_dict(), file_name)
        except Exception as err:
            print(f'Error saving model {err}')

    def save_results(self):
        file_name = self.log_file_path.replace('.log', '.pkl')
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
