# Training an agent to play River Raid with Reinforcement Learning (RL)!

This repository explores reinforcement learning algorithms and optimization techniques which can be applied to train a virtual agent to play the Atari Game Riverraid: https://en.wikipedia.org/wiki/River_Raid

Riverraid is a "shoot 'em up" game from the 80s. The player controls a fighter jet over the "River of No Return" in a raid behind enemy lines. The goal is to navigate the flight by destroying enemy tankers, helicopters, fuel depots and bridges without running out of fuel or crashing.

![A26_01](https://github.com/user-attachments/assets/cf73aaf9-3937-4530-bab1-887f257b2182)

### Initial approach

Applying a Deep Q-Network (DQN) algorithm, which uses deep neural networks to approximate the Q-value function, enabling agents to learn optimal policies in complex environments with discrete action spaces. 

- Using Gymnasium environment
- Using Stablebaselines3 library for DQN implementation
- Default model parameters
- Minimal tuning to reward behaviors like fuel seeking and to penalize loss of life (see custom environment wrapper)
- Training for 1mn timesteps (frames) / ~40k episodes on SageMaker AI using NVIDIA T4 GPU instance
- Logging and monitoring using wandb.ai

### Results:

**Key metric 1: Mean episode reward** measures the average game score the agent has achieved in the training episodes. It peaked initially and then declined, with a gentle improvement towards the end of training.
![W B Chart 08_04_2025, 15_37_42 (1)](https://github.com/user-attachments/assets/d80e979d-f48e-4afb-ac20-364ebef59ac5)

**Key metric 2: Mean episode length** denotes the average duration of a gameplay, until all three lives are lost. As you can see, it follows a similar curve as metric 1.
![W B Chart 08_04_2025, 15_37_49 (1)](https://github.com/user-attachments/assets/79571dd8-6ff9-410a-ac2e-dba84a9e901f)

At 100000 steps, the agent seems to be doing quite well, saving fuel, hitting targets, and avoiding collision:

![rl-video-step-100000-to-step-100200-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/e6e442be-ac2a-4ab7-8820-76b51ab2ead7)

At 1 mn steps, performance drops massively:

![rl-video-step-1000000-to-step-1000400-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/c422e04e-1b0b-4885-a6b5-3ecc5e124a9d)

Conclusion: Agent learns well during exploration phase (first 100 k timesteps) and then forgets, unable to optimize further.

Next steps: 
- Same approach, but longer exploration phase
- In case that doesn't work: Trying a more sophisticated policy (algorithm), e.g. PPO.


References:
https://arxiv.org/abs/1312.5602


