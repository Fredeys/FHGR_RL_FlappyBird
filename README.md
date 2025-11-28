# Reinforcement Learning for Flappy Bird  
This repository contains a complete reinforcement learning setup for a custom Flappy Bird environment.  
All components - the environment, the agent, the training pipeline, the visualisation, and the experimental analysis - were implemented from scratch to provide a transparent and reproducible RL workflow.


<p align="center">
  <img src="flappy-rl/assets/FlappyBird_Banner.gif" width="40%" alt="Flappy Bird demo banner">
</p>

The project focuses on understanding tabular Q-learning in a dynamic control task with a discretised state space.  
The environment models essential game mechanics (gravity, pipe movement, collision logic), and the agent learns via ε-greedy Q-learning across several thousand episodes.

---


## 1. Project Overview

The goal is to evaluate whether a simple tabular Q-learning agent can learn a stable policy in a sequential decision problem.  
Flappy Bird is well suited for RL because it offers:

- a minimal action space (flap / no flap),
- continuous dynamics requiring discretisation,
- immediate termination on mistakes,
- stochastic variation in pipe heights.

The project includes:

- a fully custom RL environment  
- training + evaluation pipeline  
- Pygame visualisation  
- learning curve generation  
- hyperparameter ablation  
- policy heatmap  
- evaluation score distribution  
- academic report (PDF)

---

## 2. Repository Structure

```text
flappy-rl/
│
├── src/
│   ├── env.py               # Environment: physics, states, rewards
│   ├── agent.py             # Tabular Q-learning agent
│   ├── train.py             # Training loop
│   ├── eval.py              # Evaluation mode (no exploration)
│   ├── visualizer.py        # Pygame rendering + animations
│   ├── plot_rewards.py      # Learning-curve generation
│   ├── ablation.py          # Hyperparameter experiments
│   ├── network.py           # Placeholder for DQN extension
│   └── replay_buffer.py     # Placeholder for replay buffer
│
├── assets/                  # Game sprites (bird, pipes, background, UI)
│
├── runs/                    # Training outputs (q_table.pkl, rewards.csv)
│
├── configs/                 # YAML configs for future DQN variants
│
├── report/                  # Report (LaTeX + figures)
│
└── requirements.txt         # Dependencies
```
This modular layout separates RL logic, environment modelling, visualisation, configuration, and analysis.
It supports reproducibility and simplifies future extensions.

## 3. Installation
```pip install -r requirements.txt```
Dependencies:
- **numpy** - numerical computations
- **pygame** - rendering
- **matplotlib** - plots
- **tqdm** - progress bars

## 4. Training
```python -m src.train```
Outputs are stored in runs/:
- rewards.csv - return per episode
- q_table.pkl - final Q-table

## 5. Evaluation
Run the agent without exploration:
```python -m src.eval```
A Pygame window opens.
The agent always selects the highest Q-action in each state.

## 6. Learning Curve
Generate the learning curve:
```python -m src.plot_rewards```
Saved to:
```report/figures/learning_curve.png```

## 7. Ablation Studies
Run hyperparameter ablation:
```python -m src.ablation```
Outputs:
- ablation_alpha.png
- ablation_epsilon.png
- policy_heatmap.png
- eval_score_distribution.png

Ablation varies:
- learning rate α
- epsilon decay
- discount factor γ
These help illustrate sensitivity of the Q-learning process

## 8. Method Notes
### State Discretisation
The continous state (vertical distance, horizontal distance, velocity) is mapped into bins.
This enanbles tabular RL but reduces precision.

### Reward Design
- +1 for every timestep alive
- -100 for collisions (encourages long-term survival)

### Exploration Strategy
ε-greedy during training, no exploration during evaluation.

### MDP Structure
Environment followss an MDP structure with deterministic transitions except for random pip heights.

### Limitations 
- Tabular Q-learning does not scale to large or continuous state spaces.
- Discretisation introduces information loss.
- High stochasticity would destabilize Q-value updates.
- No pixel input; agent operates on abstracted states.

## 10. Future Work
Potential extensions:
- Deep Q-Network (DQN)
- Pixel input + CNN
- Reward shaping
- Double / Dueling DQN
- Prioritized replay
- Alternative discretisation schemes
- Visual themes (night mode, dark mode, etc.)

## 11. Documentation
The full academic report is included.

## 12. Author
**Frédéric C. Kurbel**

FH Graubünden — Computational & Data Science (2025)








