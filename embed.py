import tensorflow as tf
import config as cf
import utils

with tf.name_scope('data'):
    sample_target_ids = {'sku': tf.placeholder(tf.int32, [None, ], name='sku_target_ids'),
        'video': tf.placeholder(tf.int32, [None, ], name='video_target_ids')}
    sample_history_ids = {'sku': tf.placeholder(tf.int32, [None, cf.history_len], name='sku_history_ids'),
        'video': tf.placeholder(tf.int32, [None, cf.history_len], name='video_history_ids')}
    id_embed_table = {'sku': tf.placeholder(tf.float32, [None, cf.embed_dim], name='sku_embed_table'),
        'video': tf.placeholder(tf.float32, [None, cf.embed_dim], name='video_embed_table')}


class EmbedNet(object):
    def __init__(self, sids, data_type):
        target_ids = tf.gather(sample_target_ids[data_type], sids)  # [None, ]
        history_ids = tf.gather(sample_history_ids[data_type], sids)  # [None, history_len]

        target_embeds = tf.gather(id_embed_table[data_type], target_ids)  #[None, embed_dim]
        history_embeds = tf.gather(id_embed_table[data_type], history_ids)  #[None, history_len, embed_dim]

        target_embeds = utils.dense_embedding(
            target_embeds, hidden_size=cf.rnn_size, layer_num=3, name='target_embeds')  # [None, rnn_size]
        history_embeds = self.gru_embedding(history_embeds, cf.rnn_size)
        history_embeds = tf.reshape(history_embeds, [-1, cf.history_len, cf.rnn_size])  # [None, rnn_size]
        
        # att_embeds = self.att_embedding(gru_states, item_embeds, cf.rnn_size)
        # att_embeds = gru_states[:, -1, :]  # for comparing (with or without attention)
        self.embeds = tf.concat([target_embeds, history_embeds], axis=-1)  # [None, rnn_size * 2]

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
        multi_cell = tf.contrib.rnn.MultiRNNCell(tf.contrib.rnn.GRUCell(rnn_size))
        _, gru_state = tf.nn.dynamic_rnn(multi_cell, inputs, dtype=tf.float32)
        return tf.squeeze(gru_state)

    def gru_embedding_test(self, inputs, rnn_size):
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
        return gru_states
    