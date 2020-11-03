import gym
import numpy as np
import tensorflow as tf
from policy_value_net import policy_value_net
import matplotlib.pyplot as plt

num_episodes = int(1e3)

env = gym.make('CartPole-v0')
env.seed(0)
ob_space = env.observation_space

model = policy_value_net(env)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'trained_model/model.ckpt')

    obs = env.reset()
    rewards_tracker= []

    for iteration in range(num_episodes):
        rewards = []
        while True:
            obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
            act, _ = model.act(obs=obs, stochastic=False)

            act = np.asscalar(act)
            next_obs, reward, done, _ = env.step(act)

            rewards.append(reward)

            if done:
                obs = env.reset()
                break
            else:
                obs = next_obs

        rewards_tracker.append(sum(rewards))

        smoothed_rewards_tracker = [np.mean(rewards_tracker[max(0, i - 10):i + 1]) for i in range(len(rewards_tracker))]
        plt.figure(figsize=(8, 6))
        plt.plot(rewards_tracker, 'b.', label='mean total rewards')
        plt.plot(smoothed_rewards_tracker, 'r', label='smoothed')
        plt.xlabel('episode number', fontsize=20)
        plt.legend(loc='lower right', prop={'size': 20})
        plt.savefig('test_rewards.png')
        plt.close()

        print(iteration, sum(rewards))
