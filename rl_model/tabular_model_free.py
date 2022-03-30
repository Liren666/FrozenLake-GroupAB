import numpy as np

from rl_model.tabular_model_based import policy_evaluation
import random

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        terminal = False
        a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)

        # Select action a for state s according to an e-greedy policy based on Q. by random
        while not terminal:
            next_s, r, terminal = env.step(a)
            next_a = e_greedy(q[next_s], epsilon[i], env.n_actions, random_state)
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s
            a = next_a

        if find_episodes:
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=100)
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + str(i))
                return q.argmax(axis=1), value_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    for i in range(max_episodes):
        s = env.reset()
        terminal = False
        while not terminal:
            a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)
            next_s, r, terminal = env.step(a)
            next_a = np.argmax(q[next_s])
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s

        if find_episodes:
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=100)
            if all(abs(optimal_value[i]-value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + str(i))
                return q.argmax(axis=1), value_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def e_greedy(q, epsilon, n_actions, random_state):
    if random.uniform(0, 1) < epsilon:
        a = random_state.choice(np.flatnonzero(q == q.max()))
    else:
        a = random_state.randint(n_actions)
    return a