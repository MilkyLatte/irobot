# irobot

Deep Reinforcement learning on Atari 2600 games including Pong, Breakout and Pacman

- NewDeepQTorch.py  - DQN
- DDQN.py           - double Q, DQN
- DuelingDDQN.py    - duelling DDQN
- PrioReplay_DDQN   - DDQN + priority replay

All files generate a log directory that contains values of hyperparameters, records of results after each episode and saved model weights and saved game videos every (say) 100 episodes.

From these log files the analysis.ipynb notebook can be used to create visualisations including learning curves, frame by frame Q agreement plots and more.
