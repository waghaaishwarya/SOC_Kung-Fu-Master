import numpy as np
from helper import build_model

class DQNAgent:

    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size

        self.gamma = 0.99
        self.epsilon = 1 # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.model = build_model(state_shape, action_size)
        self.target_model = build_model(state_shape, action_size)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose = 0)
        return np.argmax(q_values[0])

    def train(self, batch, batch_size):
        states, actions, rewards, next_states, dones = batch

        # predict q-values
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.argmax(next_q_values[i])

            q_values[i][actions[i]] = target

        self.model.fit(states, q_values, epochs = 1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
