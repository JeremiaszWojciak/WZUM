import gym
import time
import random

import numpy as np


def example():
    env = gym.make('CartPole-v1', render_mode='human')  # create environment
    env.reset()  # reset environment
    for _ in range(1000):  # simulation steps
        env.render()  # render image
        action = env.action_space.sample()  # choose action
        env.step(action)  # perform action
    env.close()  # close environment


def ex_1_3():
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset()
    action = 0
    for _ in range(1000):
        env.render()
        observation, reward, terminated, truncated, info = env.step(action)
        print('Pole angle: ', observation[2], 'Reward: ', reward, 'Terminated: ', terminated)
        if observation[2] > 0.0:
            action = 1
        else:
            action = 0
        if terminated:
            env.reset()
    env.close()


def ex_4_7():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4")
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    lr = 0.3
    discount_factor = 0.99
    epsilon = 0.9

    no_training_episodes = 10000
    for i in range(no_training_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[observation])
            next_observation, reward, terminated, truncated, info = env.step(action)
            max_next_observation = np.max(Q[next_observation])
            Q[observation, action] = (1 - lr) * Q[observation, action] + lr * (
                        reward + discount_factor * max_next_observation)
            observation = next_observation
            if terminated:
                done = True

    no_test_episodes = 100
    total_reward = 0
    for i in range(no_test_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[observation])
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated:
                done = True
        total_reward += episode_reward
    print(f'score: {total_reward / no_test_episodes}')
    env.close()
    print(f'Q table: {Q}')


def ex_8():
    env = gym.make("Taxi-v3")
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    lr = 0.3
    discount_factor = 0.95
    epsilon = 0.9

    no_training_episodes = 10000
    for i in range(no_training_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[observation])
            next_observation, reward, terminated, truncated, info = env.step(action)
            max_next_observation = np.max(Q[next_observation])
            Q[observation, action] = (1 - lr) * Q[observation, action] + lr * (
                    reward + discount_factor * max_next_observation)
            observation = next_observation
            if terminated:
                done = True

    env = gym.make("Taxi-v3")
    no_test_episodes = 100
    total_reward = 0
    for i in range(no_test_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[observation])
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated:
                done = True
        total_reward += episode_reward
    print(f'score: {total_reward / no_test_episodes}')
    env.close()


if __name__ == '__main__':
    ex_4_7()
