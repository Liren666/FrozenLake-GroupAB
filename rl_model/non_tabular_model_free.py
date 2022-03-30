import numpy as np

import random

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        a = e_greedy(q, epsilon[i], env.n_actions, random_state)
        terminal = False
        while not terminal:
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)
            a_new = e_greedy(q, epsilon[i], env.n_actions, random_state)

            delta = delta + (gamma * max(q))
            theta = theta + eta[i] * delta * features[a]
            features = next_s
            a = a_new

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        terminal = False

        while not terminal:
            a = e_greedy(q, epsilon[i], env.n_actions, random_state)
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)
            delta = delta + (gamma * max(q))
            theta = theta + (eta[i] * delta * features[a])
            features = next_s
    return theta


def e_greedy(q, epsilon, n_actions, random_state):
    if random.uniform(0, 1) < epsilon:
        a = random_state.choice(np.flatnonzero(q == q.max()))
    else:
        a = random_state.randint(n_actions)
    return a
