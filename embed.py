import tensorflow as tf
import config as cf

with tf.name_scope('data'):
    sample_item_ids = {'pos': tf.placeholder(tf.int32, [None, ], name='pos_item_ids'),
        'neg': tf.placeholder(tf.int32, [None, ], name='neg_item_ids')}
    sample_history_ids = {'pos': tf.placeholder(tf.int32, [None, cf.history_len], name='pos_history_ids'),
        'neg': tf.placeholder(tf.int32, [None, cf.history_len], name='neg_history_ids')}
    sample_embeds = {'pos': tf.placeholder(tf.float32, [None, cf.sample_embed_dim], name='pos_sample_embeds'),
        'neg': tf.placeholder(tf.float32, [None, cf.sample_embed_dim], name='neg_sample_embeds')}
    all_item_embeds = tf.placeholder(tf.float32, [None, cf.embed_dim], name='all_item_embeds')
    all_item_cids = tf.placeholder(tf.int32, [None, ], name='all_item_cids')


class EmbedNet(object):
    def __init__(self, item_sids, history_sids, sample_type, ns_type=None):
        item_ids = tf.gather(sample_item_ids[sample_type], item_sids)  # [None, ]
        history_ids = tf.gather(sample_history_ids[sample_type], history_sids)  # [None, history_len]
        group_ids = tf.concat([history_ids, item_ids[:, tf.newaxis]], axis=1)  # [None, history_len + 1]
        group_embeds = tf.gather(all_item_embeds, group_ids)  # [None, history_len + 1, embed_dim]

        group_cids = tf.gather(all_item_cids, group_ids)  # [None, history_len + 1]
        group_cids = tf.one_hot(group_cids, cf.one_hot_len, axis=2)  # [None, history_len + 1, one_hot_len]
        group_cids = tf.reshape(group_cids, [-1, cf.one_hot_len])
        group_cids_embeds = tf.layers.dense(group_cids, cf.cid_embed_dim, tf.nn.relu, name='cid_embeds_layer')
        group_cids_embeds = tf.reshape(group_cids_embeds, [-1, (cf.history_len + 1), cf.cid_embed_dim])
        # [None, history_len + 1, cid_embed_dim]

        group_embeds = tf.concat([tf.cast(group_embeds, tf.float32), group_cids_embeds], axis=2)
        # [None, history_len + 1, embed_dim + cid_embed_dim]
        history_embeds, item_embeds = group_embeds[:, :-1, :], group_embeds[:, -1, :]
        # [None, history_len, embed_dim + cid_embed_dim], # [None, embed_dim + cid_embed_dim]

        _, gru_states = self.gru_embedding(history_embeds, cf.rnn_size)
        gru_states = tf.reshape(gru_states, [-1, cf.history_len, cf.rnn_size])
        item_embeds = self.dense_embedding(item_embeds, hidden_size=cf.rnn_size, layer_num=3, name='item_embeds')
        
        att_embeds = self.att_embedding(gru_states, item_embeds, cf.rnn_size)
        # att_embeds = gru_states[:, -1, :]  # for comparing (with or without attention)

        if ns_type == 'ens':
            sample_embeds_ = tf.zeros(shape=[tf.shape(item_embeds)[0], cf.sample_embed_dim])
        else:
            sample_embeds_ = tf.gather(sample_embeds[sample_type], item_sids)  # [None, sample_embed_dim]
            sample_embeds_ = tf.layers.dense(sample_embeds_, cf.sample_embed_dim, tf.nn.relu, name='sample_embeds_layer')

        self.history_embeds = att_embeds
        self.item_embeds = item_embeds  # [None, rnn_size]
        self.embeds = tf.concat([att_embeds, item_embeds, sample_embeds_], axis=-1)
        self.scores = tf.squeeze(tf.layers.dense(self.embeds, 1, None, name='scores'))

    def att_embedding(self, gru_states, item_embeds, rnn_size):
        plus = True
        seq_rnn_size = rnn_size * cf.history_len
        att_inputs = tf.reshape(gru_states, shape=[-1, seq_rnn_size])
        with tf.variable_scope('attention', initializer=tf.random_normal_initializer(mean=0., stddev=1.)):
            att_W_h = tf.get_variable(name='W_h', shape=[seq_rnn_size, seq_rnn_size])
            att_W_i = tf.get_variable(name='W_i', shape=[rnn_size, rnn_size])
            if plus:
                att_v = tf.get_variable(name='v', shape=[seq_rnn_size, seq_rnn_size])
            else:
                att_v = tf.get_variable(name='v', shape=[seq_rnn_size * 2, seq_rnn_size])
        tmp_h = tf.matmul(att_inputs, att_W_h)  # [None, seq_rnn_size]
        tmp_i = tf.matmul(item_embeds, att_W_i)  # [None, rnn_size]
        tmp_i = tf.concat([tmp_i] * cf.history_len, axis=1)  # [None, seq_rnn_size]
        if plus:
            att_hidden = tf.tanh(tf.add(tmp_h, tmp_i))
        else:
            att_hidden = tf.tanh(tf.concat([tmp_h, tmp_i], axis=1))
        e_i = tf.matmul(att_hidden, att_v)  # [None, 1]
        e_i = tf.reshape(e_i, shape=[-1, cf.history_len, rnn_size])
        alpha_i = tf.nn.softmax(e_i, axis=1)
        c_i = tf.multiply(alpha_i, gru_states)
        c = tf.reduce_sum(c_i, axis=1)
        return c

    def gru_embedding(self, inputs, rnn_size):
        cell = tf.contrib.rnn.GRUCell(rnn_size)
        batch_size = tf.shape(inputs)[0]
        states, outputs = [cell.zero_state(batch_size, tf.float32)], []
        with tf.variable_scope('gru') as scope:
            for i in range(cf.history_len):
                if i > 0:
                    scope.reuse_variables()
                output, state = cell(inputs[:, i, :], states[-1])
                outputs.append(output)
                states.append(state)
        gru_outputs, gru_states = tf.stack(outputs), tf.stack(states[1:])
        # gru_outputs == gru_states
        return gru_outputs, gru_states

    def dense_embedding(self, inputs, hidden_size, layer_num, name):
        state = inputs
        for layer in range(layer_num):
            activation = tf.nn.relu if layer < layer_num - 1 else tf.nn.tanh
            layer_name = name + '_layer%i' % layer
            state = tf.layers.dense(state, hidden_size, activation, name=layer_name)
        return state