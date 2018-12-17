import tensorflow as tf
import config as cf
from embed import EmbedNet


class Discriminator(object):
    def __init__(self):
        with tf.name_scope('dis_input'):
            self.pos_sids = tf.placeholder(tf.int32, [None, ], name='pos_sids')
            self.neg_item_sids = tf.placeholder(tf.int32, [None, ], name='neg_item_sids')
            self.neg_history_sids = tf.placeholder(tf.int32, [None, ], name='neg_history_sids')
            self.labels = tf.placeholder(tf.float32, [None, ], name='labels')

            self.lr = tf.Variable(cf.dis_lr, trainable=False, dtype=tf.float32)
            self.new_lr = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.new_lr)

        with tf.variable_scope('dis_embed_net', initializer=cf.initializer) as scope:
            pos_net = EmbedNet(self.pos_sids, self.pos_sids, 'pos')
            scope.reuse_variables()
            neg_net = EmbedNet(self.neg_item_sids, self.neg_history_sids, 'neg')
            self.scores = {'pos': pos_net.scores, 'neg': neg_net.scores}

        with tf.name_scope('dis_optimize'):
            if cf.dis_loss == 'hinge':
                self.loss = tf.reduce_sum(tf.maximum(0., self.scores['neg'] - self.scores['pos'] + cf.margin))
            else:  # cf.dis_loss == 'ce'
                self.loss = tf.losses.sigmoid_cross_entropy(
                    tf.constant([1.] * cf.batch_size + [0.] * cf.batch_size * cf.neg_k),
                    tf.concat([self.scores['pos'], self.scores['neg']], axis=0))
                self.test_loss = {
                    'pos': tf.losses.sigmoid_cross_entropy(self.labels, self.scores['pos']),
                    'neg': tf.losses.sigmoid_cross_entropy(self.labels, self.scores['neg'])}
            tf.summary.scalar('dis_loss', self.loss, collections=['dis'])
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('rewards'):
            pos_item_embeds = tf.concat([pos_net.item_embeds] * cf.neg_k, axis=0)
            neg_item_embeds = neg_net.item_embeds
            item_distance = tf.norm(tf.subtract(pos_item_embeds, neg_item_embeds), axis=1)
            pos_history_embeds = tf.concat([pos_net.history_embeds] * cf.neg_k, axis=0)
            neg_history_embeds = neg_net.history_embeds
            history_distance = tf.norm(tf.subtract(pos_history_embeds, neg_history_embeds), axis=1)
            distance = cf.item_lambda * item_distance + cf.history_lambda * history_distance  # [batch_size]
            self.rewards = self.scores['neg'] - distance

        self.summary_op = tf.summary.merge_all(key='dis')
