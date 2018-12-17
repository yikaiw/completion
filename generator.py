import tensorflow as tf
import config as cf
from embed import EmbedNet


class Generator(object):
    def __init__(self, ns_type):
        with tf.name_scope('gen_input_' + ns_type):
            self.pos_sids = tf.placeholder(tf.int32, [None, ], name='pos_sids')
            self.neg_cand_sids = tf.placeholder(tf.int32, [None, cf.cs], name='neg_cand_sids')
            self.neg_cand_idxs = tf.placeholder(tf.int32, [None, 2], name='neg_cand_idxs')
            self.rewards = tf.placeholder(tf.float32, [None, ], name='rewards')
            self.baseline = tf.placeholder(tf.float32, name='rewards_baseline')
            self.tem = tf.placeholder(tf.float32, name='temperature')

            self.lr = tf.Variable(cf.gen_lr, trainable=False, dtype=tf.float32)
            self.new_lr = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.new_lr)

        with tf.variable_scope('gen_embed_net_' + ns_type, initializer=cf.initializer) as scope:
            pos_net = EmbedNet(self.pos_sids, self.pos_sids, 'pos')
            scope.reuse_variables()
            neg_cand_sids = tf.reshape(self.neg_cand_sids, [-1])
            if ns_type == 'ons':
                neg_cand_net = EmbedNet(neg_cand_sids, neg_cand_sids, 'neg', ns_type)
            else:  # ns_type == 'ens':
                pos_sids = tf.reshape(tf.concat([self.pos_sids[:, tf.newaxis]] * cf.cs, axis=1), [-1])
                neg_cand_net = EmbedNet(pos_sids, neg_cand_sids, 'neg', ns_type)

        with tf.name_scope('gen_neg_sampling_' + ns_type):
            pos_embeds = pos_net.embeds  # [batch_size, gen_embed_dim]
            pos_embeds_cs = tf.concat([tf.concat([pos_embeds[:, tf.newaxis, :]] * cf.cs, axis=1)] * cf.neg_k, axis=0)
            neg_cand_embeds = tf.reshape(neg_cand_net.embeds, [-1, cf.cs, cf.embed_concat_size])
            # pos_embeds_cs, neg_cand_embeds: [batch_size * neg_k, cs, gen_embed_dim]
            if cf.gen_distance_metric == 'inner_prod':
                self.inner_prod = tf.reduce_sum(tf.multiply(neg_cand_embeds, pos_embeds_cs), axis=2)
                self.neg_cand_probs = tf.nn.softmax(self.inner_prod / self.tem, axis=-1)
                # self.inner_prod, self.neg_cand_probs: [batch_size * neg_k, cs]
            elif cf.gen_distance_metric == 'nor_inner_prod':
                # embeds_norm = tf.multiply(tf.norm(neg_cand_embeds, axis=2), tf.norm(neg_cand_embeds, axis=2))
                embeds_norm = tf.norm(neg_cand_embeds, axis=2)
                inner_prod = tf.reduce_sum(tf.multiply(neg_cand_embeds, pos_embeds_cs), axis=2)
                self.nor_inner_prod = tf.div(inner_prod, embeds_norm)
                self.neg_cand_probs = tf.nn.softmax(self.nor_inner_prod / self.tem, axis=-1)
                # self.inner_prod, self.neg_cand_probs: [batch_size * neg_k, cs]
            else:  # cf.gen_distance_metric == 'mse'
                self.distance = tf.reduce_sum(tf.square(tf.subtract(neg_cand_embeds, pos_embeds_cs)), axis=2)
                self.neg_cand_probs = tf.nn.softmax(-self.distance / self.tem, axis=-1)
                # self.distance, self.neg_cand_probs: [batch_size * neg_k, cs]

        with tf.name_scope('gen_optimize_' + ns_type):
            neg_probs = tf.gather_nd(self.neg_cand_probs, self.neg_cand_idxs)  # [batch_size * neg_k]
            # utils.variable_summaries(neg_probs, collections='gen')
            self.loss = -tf.reduce_sum(tf.multiply(self.rewards - self.baseline, tf.log(neg_probs + 1e-5)))
            tf.summary.scalar('gen_loss', self.loss, collections=['gen_' + ns_type])
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.summary_op = tf.summary.merge_all(key='gen_' + ns_type)
