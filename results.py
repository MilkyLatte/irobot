from datetime import datetime
import os


class results():
    def __init__(self, global_vars_dict : dict):
        super().__init__()
        # create a log file in the log directory
        self.time_start = datetime.now()
        self.log_file_name = 'results_' + self.time_start.strftime('%Y%m%d_%H%M') + ".log"
        self.log_file_path = os.path.join('logs', self.log_file_name)
        if not os.path.exists(os.path.dirname(self.log_file_path)):
            os.makedirs(os.path.dirname(self.log_file_path))
        self.file_handle = f = open(self.log_file_path, mode='w')

        # write a brief descriptor 
        f.write('Train started: ' + self.time_start.strftime('%H:%M') + '\n')
        f.writelines(f'{k} = {v}\n' for k, v in global_vars_dict.items() if k==k.upper())
        f.write('\ngames,frames,epsilon,score,network_loss,delta_t,frames/t\n')
        # self.best_score = None
        self.last_t = self.time_start
        self.last_frames = 0

        
    def record(self, frames, games, epsilon, score, network_loss):
        t = datetime.now()
        delta_t = (t - self.last_t).total_seconds()
        delta_frames = (frames - self.last_frames)
        self.last_t = t
        self.last_frames = frames
        # self.best_score = max(score, self.best_score)
        self.file_handle.write(
            t.strftime('%H:%M:%S') + f',{games},{frames},{epsilon},{score},{network_loss},{delta_t},{delta_frames/delta_t}\n')
        self.file_handle.flush()

        

    def close(self):
        self.file_handle.close()

