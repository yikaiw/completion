import os
from os.path import join
import argparse
import tensorflow as tf
from datetime import datetime
note = datetime.now().strftime('%Y%m%d-%H%M')

random_seed = 25
initializer = tf.contrib.layers.xavier_initializer(seed=random_seed)

id_num, embed_num = None, None
sample_num = {'train': None, 'test': None}
class_labels, sample_idx = None, None
test_subsample_num = {'pos': 1000, 'neg': None}
neg_sid_noise = None
test_all_data, save_model, denoise = True, False, False
ns_ratio = 1.  # ons: observed negative samples, ens: expanded negative samples

result_dir = 'results'
data_dir = '../data7'
sample_dirs = {'train': ['0705', '0706'], 'test': ['0707']}

batch_size, test_batch_size = 1024, 40000
# batch_size, test_batch_size = 512, 8192
epoch_num, epoch_start_gen, round_num = 50, 3, 5

neg_k = 1  # int, neg: pos

lr = 0.01
lr_decay_epoch, lr_decay = 10, 0.5
class_lam = 0.1
embed_dim, history_len, sample_embed_dim = 50, 10, 8
dis_hidden_dim, gen_embed_dim = 30, 30
cid_embed_dim = 8
margin = 1
tem, tem_decay = 20, 1.00

rnn_size = 58
rnn_layers = 1
fc_layers = 4
attn_length = 10

embed_concat_size = rnn_size * 2 + sample_embed_dim

dirs = [result_dir, join(data_dir, 'train'), join(data_dir, 'test')]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str, help='Root directory path of data')
    parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    
    return parser.parse_args()