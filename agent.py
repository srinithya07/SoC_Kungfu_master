import numpy as np
import random
from helper import KungFu
from memory import ReplayBuffer


class DQNAgent:
    def __init__(self, state_shape, action_space, learn_rate=0.00025, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.998):
        self.state_shape = state_shape
        self.possible_actions = list(range(action_space))
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = KungFu(self)
        self.target_model = KungFu(self)
        self.update_target_network()
        self.memory = ReplayBuffer(capacity=100000)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.possible_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size=32):
        if len(self.memory) < 300:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size)
        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next[i])
            q_values[i][actions[i]] = target
        self.model.fit(states, q_values, epochs=1, verbose=0)
        