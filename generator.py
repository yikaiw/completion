import tensorflow as tf
import config as cf
from embed import EmbedNet
import utils


class Generator(object):
    def __init__(self):
        with tf.name_scope('input'):
            self.sids = tf.placeholder(tf.int32, [None, ], name='sids')
            self.class_labels = tf.placeholder(tf.float32, [None, ], name='labels')  # labels indicate pos or neg

            self.lr = tf.Variable(cf.lr, trainable=False, dtype=tf.float32)
            self.new_lr = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.new_lr)

        with tf.variable_scope('embed_net', initializer=cf.initializer):
            sku_net = EmbedNet(self.sids, modality_type='sku')
            video_net = EmbedNet(self.sids, modality_type='video')

            sku_embed, real_video_embed = sku_net.embed, video_net.embed

        with tf.variable_scope('generator_net', initializer=cf.initializer):
            fake_video_embed = utils.dense_embedding(
                sku_embed, hidden_size=cf.rnn_size, layer_num=3, name='fake_video_embed')

            real_embed = tf.concat([sku_embed, real_video_embed], axis=1)
            fake_embed = tf.concat([sku_embed, fake_video_embed], axis=1)

        with tf.variable_scope('critic_score_net', initializer=cf.initializer) as scope:
            self.real_critic_scores = self.score_net(real_embed, name='critic')
            scope.reuse_variables()
            self.fake_critic_scores = self.score_net(fake_embed, name='critic')

        with tf.variable_scope('class_score_net', initializer=cf.initializer) as scope:
            self.real_class_scores = self.score_net(real_embed, name='class')
            scope.reuse_variables()
            self.fake_class_scores = self.score_net(fake_embed, name='class')

        with tf.name_scope('optimize'):
            self.dis_loss = tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.real_critic_scores), self.real_critic_scores) \
                + tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.real_critic_scores), self.fake_critic_scores)
            self.gen_loss = tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.fake_critic_scores), self.fake_critic_scores)

            self.class_loss = tf.losses.sigmoid_cross_entropy(self.class_labels, self.real_class_scores) + \
                tf.losses.sigmoid_cross_entropy(self.class_labels, self.fake_class_scores)

            self.loss = self.dis_loss + self.gen_loss + cf.class_lam * self.class_loss
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def score_net(self, inputs, name):
        states = utils.dense_embedding(
            inputs, hidden_size=cf.rnn_size, layer_num=2, name='%s_score_net' % name)
        scores = tf.squeeze(tf.layers.dense(states, 1, None, name='%s_scores' % name))
        return scores
