import numpy as np

################ Model-based algorithms ################

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float64)

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum(
                [env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in
                 range(env.n_states)])

            delta = max(delta, abs(v - value[s]))
        if delta < theta:
            break
    return value


def policy_improvement(env, value, gamma, policy):
    policy_stable = True
    for s in range(env.n_states):
        pol = policy[s].copy()
        v = [
            sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
            for a in range(env.n_actions)]
        policy[s] = np.argmax(v)
        if pol != policy[s]:
            policy_stable = False
    return policy_stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states, dtype=int)
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    policy_stable = False
    index = 0
    while not policy_stable:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy_stable = policy_improvement(env, value, gamma, policy)
        index += 1
    print(index)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    index = 0
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    for _ in range(max_iterations):
        delta = 0.
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum(
                [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
                for a in range(env.n_actions)])

            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            break

        index = index + 1

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax([sum(
            [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for
            a in range(env.n_actions)])

    print(index)
    return policy, value