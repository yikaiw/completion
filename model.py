import tensorflow as tf
import config as cf
from embed import EmbedNet
import utils


class GAN(object):
    def __init__(self):
        with tf.name_scope('input'):
            self.sids = tf.placeholder(tf.int32, [None, ], name='sids')
            self.labels = tf.placeholder(tf.float32, [None, ], name='labels')

            self.lr = tf.Variable(cf.dis_lr, trainable=False, dtype=tf.float32)
            self.new_lr = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.new_lr)

        with tf.variable_scope('embed_net', initializer=cf.initializer):
            sku_net = EmbedNet(self.sids, data_type='sku')
            video_net = EmbedNet(self.sids, data_type='video')

            sku_embed, real_video_embed = sku_net.embed, video_net.embed
            fake_video_embed = utils.dense_embedding(
                sku_embed, hidden_size=cf.rnn_size, layer_num=3, name='fake_video_embed')

            real_embed = tf.concat([sku_embed, real_video_embed], axis=1)
            fake_embed = tf.concat([sku_embed, fake_video_embed], axis=1)

        with tf.variable_scope('gan_score_net', initializer=cf.initializer) as scope:
            self.real_gan_scores = self.score_net(real_embed, name='gan')
            scope.reuse_variables()
            self.fake_gan_scores = self.score_net(fake_embed, name='gan')

        with tf.variable_scope('class_score_net', initializer=cf.initializer) as scope:
            self.real_class_scores = self.score_net(real_embed, name='class')
            scope.reuse_variables()
            self.fake_class_scores = self.score_net(fake_embed, name='class')

        with tf.name_scope('dis_optimize'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.labels, self.scores),
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def score_net(self, inputs, name):
        states = utils.dense_embedding(
            inputs, hidden_size=cf.rnn_size, layer_num=2, name='%s_score_net' % name)
        scores = tf.squeeze(tf.layers.dense(states, 1, None, name='%s_scores' % name))
        return scores
