import tensorflow as tf
import numpy as np

class reinforce_train:
    def __init__(self, model, gamma=0.95, c_1=1):
        """
        inputs-
        model: policy-value network
        gamma: discount factor
        param c_1: coefficient for value difference term of loss
        """

        self.model = model
        self.gamma = gamma

        self.pi_trainable = self.model.get_trainable_variables()

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.values = tf.placeholder(dtype=tf.float32, shape=[None], name='values')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        # probabilities of actions which agent took with policy
        act_probs = self.model.act_probs
        selected_action_prob = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        selected_action_prob = tf.reduce_sum(selected_action_prob, axis=1)

        with tf.variable_scope('loss/reinforce'):
            loss_reinforce = -tf.reduce_mean(tf.log(selected_action_prob) * self.gaes)

        with tf.variable_scope('loss/vf'):
            v_preds = self.model.v_preds
            loss_vf = tf.squared_difference(self.values, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)

        with tf.variable_scope('loss'):
            loss = loss_reinforce + c_1 * loss_vf

        # Get gradients and variables
        self.gradient_holder = []
        for j, var in enumerate(self.pi_trainable):
            self.gradient_holder.append(tf.placeholder(tf.float32, name='grads' + str(j)))

        self.gradients = tf.gradients(loss, self.pi_trainable)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.apply_gradients(zip(self.gradient_holder, self.pi_trainable))

    def get_vars(self):
        net_vars = tf.get_default_session().run(self.pi_trainable)
        return net_vars

    def get_grads(self, obs, actions, values, gaes):
        grads = tf.get_default_session().run(self.gradients, feed_dict={self.model.obs: obs, self.actions: actions,
                                                                        self.values: values, self.gaes: gaes})
        return grads

    def update(self, gradient_buffer):
        feed = dict(zip(self.gradient_holder, gradient_buffer))
        tf.get_default_session().run(self.train_op, feed_dict = feed)

    def get_values(self, rewards):
        values = np.zeros(len(rewards))
        cumulative_rewards = 0
        for i in reversed(range(0, len(rewards))):
            cumulative_rewards = cumulative_rewards * self.gamma + rewards[i]
            values[i] = cumulative_rewards
        return values

    def get_gaes(self, values, v_preds):
        gaes = values - v_preds
        return gaes
