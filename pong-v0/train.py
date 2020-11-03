import gym
import numpy as np
import tensorflow as tf
from policy_value_net import policy_value_net
from reinforce import reinforce_train
import matplotlib.pyplot as plt

mini_batch_size = 8
num_episodes = int(1e6)
gamma = 0.99

env = gym.make('Pong-v0')
env.seed(0)
model = policy_value_net()
reinforce = reinforce_train(model, gamma=gamma)
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
    obs = env.reset()
    obs = preprocess(obs)
    success_num = 0
    partial_win = -20

    rewards_tracker = []
    rewards_each_mb = []

    # set up gradient buffers and set values to 0
    grad_buffer_pe = reinforce.get_vars()
    for i, g in enumerate(grad_buffer_pe):
        grad_buffer_pe[i] = g * 0

    for iteration in range(num_episodes):
        observations = []
        actions = []
        v_preds = []
        rewards = []
        while True:
            obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder model.obs
            act, v_pred = model.act(obs=obs, stochastic=True)

            act = np.asscalar(act)
            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, _ = env.step(act+2) # only learn RIGHT and LEFT

            observations.append(obs)
            actions.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)

            if done:
                obs = env.reset()
                obs = preprocess(obs)
                break
            else:
                obs = next_obs
                obs = preprocess(obs)

        if sum(rewards) >= 20:
            success_num += 1
            if success_num >= 100:
                saver.save(sess, './trained_model/model.ckpt')
                print('pong solved! model saved.')
                break
        else:
            success_num = 0

        values = reinforce.get_values(rewards)
        gaes = reinforce.get_gaes(values=values, v_preds=v_preds)

        # convert list to numpy array for feeding tf.placeholder
        observations = np.reshape(observations, newshape=[-1, 80, 80])
        actions = np.array(actions).astype(dtype=np.int32)
        values = np.array(values).astype(dtype=np.float32)
        gaes = np.array(gaes).astype(dtype=np.float32)

        pe_grads = reinforce.get_grads(obs=observations, actions=actions, values=values, gaes=gaes)
        for i, g in enumerate(pe_grads):
            grad_buffer_pe[i] += g

        rewards_tracker.append(sum(rewards))

        # update
        if iteration % mini_batch_size == 0 and iteration != 0:
            rewards_each_mb.append(np.mean(rewards_tracker))
            rewards_tracker = []

            smoothed_rewards_each_mb = [np.mean(rewards_each_mb[max(0, i - 10):i + 1]) for i in range(len(rewards_each_mb))]
            plt.figure(figsize=(8, 6))
            plt.plot(rewards_each_mb, 'b.', label='mean total rewards')
            plt.plot(smoothed_rewards_each_mb, 'r', label='smoothed')
            plt.xlabel('mini batch number', fontsize=20)
            plt.legend(loc='lower right', prop={'size': 20})
            plt.savefig('train_rewards.png')
            plt.close()

            if smoothed_rewards_each_mb[-1] >= partial_win:
                saver.save(sess, './partially_trained_model/model.ckpt')
                print('learning pong better! model saved.', partial_win)
                partial_win += 1

            reinforce.update(grad_buffer_pe)

            # clear buffer values for next mini batch
            for i, g in enumerate(grad_buffer_pe):
                grad_buffer_pe[i] = g * 0

        # print(iteration)
