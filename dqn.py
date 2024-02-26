import random

import numpy as np
import gym
from gym import wrappers


class DQN:
    def __init__(self, n_episodes=500, gamma=0.99, batch_size=32,
                 epsilon=1.0, decay=0.005, min_epsilon=0.1, memory_limit=500):
        self.memory_limit = memory_limit  # limit of experience replay memory
        self.min_epsilon = min_epsilon  # minimum value of epsilon
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate, epsilon-greedy value
        self.n_episodes = n_episodes  # number of episodes
        self.batch_size = batch_size  # batch size for experience replay
        self.decay = decay  # decay rate for epsilon

    def init_environment(self, name='CartPole-v1', monitor=False):
        self.env = gym.make(name)
        if monitor:
            self.env = wrappers.Monitor(self.env, name, force=True, video_callable=False)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.replay = []

    def init_model(self, model):
        self.model = model(self.n_actions)

    def train(self, render=False):
        max_reward = 0

        for episode in range(self.n_episodes):
            state = self.env.reset()[0]
            total_reward = 0
            while True:
                if render:
                    self.env.render()

                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(self.model.predict(state[np.newaxis, :])[0])

                # run one timestep of the environment's dynamics
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay.append([state, action, reward, new_state, done])

                # sample batch from experience replay
                batch_size = min(self.batch_size, len(self.replay))
                batch = random.sample(self.replay, batch_size)
                X = np.zeros((batch_size, self.n_states))
                y = np.zeros((batch_size, self.n_actions))

                states = np.array([x[0] for x in batch])
                new_states = np.array([x[3] for x in batch])

                Q = self.model.predict(states)
                new_Q = self.model.predict(new_states)

                # construct training data
                for i in range(batch_size):
                    state_r, action_r, reward_r, next_state_r, done_r = batch[i]
                    target = Q[i]

                    if done_r:
                        target[action_r] = reward_r
                    else:
                        target[action_r] = reward_r + self.gamma * np.amax(new_Q[i])

                    X[i, :] = state_r
                    y[i, :] = target

                # train deep learning model
                self.model.fit(X, y, batch_size=batch_size, epochs=1, verbose=0)

                total_reward += reward
                state = new_state
                if done:
                    break

            # remove old entries from replay memory
            while len(self.replay) > self.memory_limit:
                self.replay.pop(0)
            self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay * episode)

            if total_reward > max_reward:
                max_reward = total_reward
            print(
                f'Episode: {episode + 1}, Total Reward: {total_reward}, '
                f'Max Reward: {max_reward}, Epsilon: {self.epsilon}')

        print('Training finished')

    def play(self, episodes):
        for _ in range(episodes):
            state = self.env.reset()[0]
            total_reward = 0

            while True:
                self.env.render()
                action = np.argmax(self.model.predict(state[np.newaxis, :])[0])
                new_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                if done:
                    break
            print(f'Total Reward: {total_reward}')
        self.env.close()


def mlp_model(n_actions):
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(24, input_dim=4, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':

    dqn = DQN(n_episodes=2500, batch_size=64)
    dqn.init_environment("CartPole-v1")
    dqn.init_model(mlp_model)

    try:
        dqn.train(render=False)
    except KeyboardInterrupt:
        pass
    dqn.play(episodes=100)
