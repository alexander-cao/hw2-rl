import gym
import numpy as np
import tensorflow as tf
from policy_value_net import policy_value_net
from reinforce import reinforce_train
import matplotlib.pyplot as plt

mini_batch_size = 8
num_episodes = int(1e6)
gamma = 0.95

env = gym.make('CartPole-v0')
env.seed(0)
ob_space = env.observation_space
model = policy_value_net(env)
reinforce = reinforce_train(model, gamma=gamma)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    obs = env.reset()
    success_num = 0

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

            next_obs, reward, done, _ = env.step(act)

            observations.append(obs)
            actions.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)

            if done:
                obs = env.reset()
                break
            else:
                obs = next_obs

        if sum(rewards) >= 199:
            success_num += 1
            if success_num >= 100:
                saver.save(sess, './trained_model/model.ckpt')
                print('cartpole solved! model saved.')
                break
        else:
            success_num = 0

        values = reinforce.get_values(rewards)
        gaes = reinforce.get_gaes(values=values, v_preds=v_preds)

        # convert list to numpy array for feeding tf.placeholder
        observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
        actions = np.array(actions).astype(dtype=np.int32)
        values = np.array(values).astype(dtype=np.float32)
        gaes = np.array(gaes).astype(dtype=np.float32)
        gaes = (gaes - gaes.mean()) / gaes.std()

        pe_grads = reinforce.get_grads(obs=observations, actions=actions, values=values, gaes=gaes)
        for i, g in enumerate(pe_grads):
            grad_buffer_pe[i] += g

        rewards_tracker.append(sum(rewards))

        # update
        if iteration % mini_batch_size == 0 and iteration != 0:
            reinforce.update(grad_buffer_pe)

            # clear buffer values for next mini batch
            for i, g in enumerate(grad_buffer_pe):
                grad_buffer_pe[i] = g * 0

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

        print(iteration)
