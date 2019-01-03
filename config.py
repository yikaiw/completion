import tensorflow as tf
from datetime import datetime
note = datetime.now().strftime('%Y%m%d-%H%M')

random_seed = 25
initializer = tf.contrib.layers.xavier_initializer(seed=random_seed)

id_num, embed_num = None, None
sample_num = {'train': None, 'test': None}
class_labels, sample_idx = None, None

result_dir = 'results'
data_dir = '../data7'
sample_dirs = {'train': ['0705', '0706'], 'test': ['0707']}

batch_size, test_batch_size = 1024, 40000
epoch_num = 15

lr = 0.01
lr_decay_epoch, lr_decay = 10, 0.5
class_lam = 0.1
neg_k = 1  # int, neg: pos
embed_dim, history_len = 50, 10
missing_rate = 0.5

rnn_size = 50
attn_length = 10

embed_concat_size = rnn_size * 2
