import numpy as np
import random


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9997,
        min_epsilon: float = 0.05,
    ):
        # Q-Table: {state: [Q(a0), Q(a1)]}
        self.Q: dict = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_qs(self, state):
        if state not in self.Q:
            self.Q[state] = [0.0, 0.0]  # zwei Aktionen (0 = nichts, 1 = flap)
        return self.Q[state]

    def select_action(self, state) -> int:
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.get_qs(state)))

    def update(self, state, action, reward, next_state):
        qs = self.get_qs(state)
        next_qs = self.get_qs(next_state)

        target = reward + self.gamma * max(next_qs)
        qs[action] += self.alpha * (target - qs[action])

        # epsilon-Decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)