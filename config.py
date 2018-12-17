import os
from os.path import join
import tensorflow as tf
from datetime import datetime
note = datetime.now().strftime('%Y%m%d-%H%M')

random_seed = 25
initializer = tf.contrib.layers.xavier_initializer(seed=random_seed)

id_num, one_hot_len, embed_num = None, 4747, None
sample_num = {'train': {'pos': None, 'neg': None}, 'test': {'pos': None, 'neg': None}}
test_subsample_num = {'pos': 1000, 'neg': None}
neg_sid_noise = None
test_all_data, save_model, denoise = True, False, False
ns_ratio = 1.  # ons: observed negative samples, ens: expanded negative samples

param_record_dir = 'param_record'
model_dir = 'checkpoints'
log_dir = 'logs'
result_dir = 'results'
data_dir = '../data7'
sample_dirs = {'train': ['0705', '0706'], 'test': ['0707']}

batch_size, test_batch_size = 1024, 40000
# batch_size, test_batch_size = 512, 8192
epoch_num, epoch_start_gen, round_num = 50, 3, 5

dis_loss = 'hinge'  # hinge, ce
gen_distance_metric = 'nor_inner_prod'  # inner_prod, nor_inner_prod, mse
dis_distance_metric = 'mse'  # mse, cos
neg_sampling_method = 'random'  # random, random_with_item

neg_k = 1  # int, neg: pos

dis_lr, gen_lr = 0.02, 0.01
lr_decay_epoch, lr_decay = 10, 0.5
embed_dim, history_len, sample_embed_dim = 50, 10, 8
dis_hidden_dim, gen_embed_dim = 30, 30
cid_embed_dim = 8
margin = 1
cs = 20  # candidate size
tem, tem_decay = 20, 1.00

rnn_size = 58
rnn_layers = 1
fc_layers = 4
attn_length = 10

embed_concat_size = rnn_size * 2 + sample_embed_dim

dirs = [model_dir, log_dir, result_dir, join(data_dir, 'train'), join(data_dir, 'test')]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

if dis_loss == 'hinge':
    neg_k = 1

ce_loss = True if dis_loss == 'ce' else False