import tensorflow as tf

class policy_value_net:
    def __init__(self):
        with tf.variable_scope('policy_value_net'):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80], name='obs')
            obs_flattened = tf.layers.flatten(self.obs)

            with tf.variable_scope('policy_net'):
                policy_layer_1 = tf.layers.dense(inputs=obs_flattened, units=128, activation=tf.nn.elu,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.act_probs = tf.layers.dense(inputs=policy_layer_1, units=2, activation=tf.nn.softmax,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope('value_net'):
                value_layer_1 = tf.layers.dense(inputs=obs_flattened, units=128, activation=tf.nn.elu,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.v_preds = tf.layers.dense(inputs=value_layer_1, units=1, activation=None,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
