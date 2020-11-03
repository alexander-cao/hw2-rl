import gym
import numpy as np
import tensorflow as tf
from policy_value_net import policy_value_net
import matplotlib.pyplot as plt

num_episodes = int(1e3)

env = gym.make('Pong-v0')
env.seed(0)

model = policy_value_net()
saver = tf.train.Saver()

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, 'trained_model/model.ckpt')
    saver.restore(sess, 'partially_trained_model/model.ckpt')

    obs = env.reset()
    obs = preprocess(obs)
    rewards_tracker= []

    for iteration in range(num_episodes):
        rewards = []
        while True:
            obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder model.obs
            act, _ = model.act(obs=obs, stochastic=False)

            act = np.asscalar(act)
            next_obs, reward, done, _ = env.step(act+2) # only learn RIGHT and LEFT

            rewards.append(reward)

            if done:
                obs = env.reset()
                obs = preprocess(obs)
                break
            else:
                obs = next_obs
                obs = preprocess(obs)

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
