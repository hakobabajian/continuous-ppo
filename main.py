import numpy as np
from agent import Agent
from environment import Environment
from matplotlib import pyplot as plt
from timeit import default_timer
import os

if __name__ == '__main__':
    kenv = Environment(max_runtime=30,
                       target_offset=30,
                       observations=3)
    N = 20
    batch_size = 5
    n_epochs = 4

    agent = Agent(n_actions=kenv.actions,
                  batch_size=batch_size,
                  n_epochs=n_epochs,
                  input_dims=kenv.observation_space_shape,
                  entropy_coefficient=0.01)

    n_iterations = 50
    best_score = kenv.max_punishment
    score_history = []
    times = []
    altitudes = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    sets = list(int(d[d.rindex('_') + 1: d.index('.')]) for d in os.listdir('tmp/plots'))
    set = max(sets) + 1
    print(set)

    for i in range(n_iterations):
        observation = kenv.reset()
        done = False
        score = 0

        start_time = default_timer()
        times.append([])
        altitudes.append([])

        while not done:
            action, prob = agent.choose_action(np.array(observation, dtype=np.float32))
            observation_, reward, done = kenv.step(action)
            n_steps += 1

            times[-1].append(default_timer() - start_time)
            altitudes[-1].append(observation_[1])

            score += reward
            agent.remember(np.array(observation, dtype=np.float32), np.array(observation_, dtype=np.float32),
                           action, prob, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps,
              'learning_steps', learn_iters)
    agent.save_models()

    for i, run in enumerate(times):
        plt.plot(times[i], altitudes[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.title(f"Altitude Control Guidance - Deep Learning ({n_iterations} iterations)")

    ax = plt.gca()
    ax.set_xlim([0, 30])
    ax.set_ylim([60, 140])

    plt.savefig(f'tmp/plots/altitudes_set_{set}.png')
