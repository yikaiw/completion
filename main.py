import numpy as np
import tensorflow as tf
import os
import argparse
import warnings
from generator import Generator
import utils
import config as cf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', default='/root/data/ActivityNet', type=str, help='Root directory path of data')
parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
parser.add_argument('--result_path', default='results', type=str, help='Result directory path')

args = parser.parse_args()
for arg in args.__dict__:
    setattr(cf, arg, args.__dict__[arg])

warnings.filterwarnings('ignore')
tf.set_random_seed(cf.random_seed)
np.random.seed(cf.random_seed)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

print('Task name: %s' % cf.note)
print('Loading data.')
train_data_dict, test_data_dict = utils.get_data_dicts()
print('Building Generator.')
gen = Generator()

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())
step = 0

loss_list, auc_list = [], []
train_sample_num = cf.sample_num['train']['pos'] // cf.batch_size * cf.batch_size
best_auc = 0.

print('Start training:')
for epoch in range(cf.epoch_num):
    # -Train-
    print('\nFor epoch %i (/%i):' % (epoch, cf.epoch_num - 1))
    pos_idx = cf.sample_idx['train']['pos']
    permu = np.random.permutation(pos_idx)[:train_sample_num]
    lr = cf.lr * cf.lr_decay ** (epoch // cf.lr_decay_epoch)
    sess.run(gen.lr_update, feed_dict={gen.new_lr: lr})
    for sample_idx in range(0, train_sample_num, cf.batch_size):
        pos_sids = permu[sample_idx: sample_idx + cf.batch_size]
        neg_sids = utils.get_negs(cf.batch_size, cf.neg_k)  # [batch_size * neg_k]
        sids = np.concatenate([pos_sids, neg_sids])
        class_labels = np.concatenate(
            [np.ones_like(pos_sids, dtype=np.float32), np.zeros_like(neg_sids, dtype=np.float32)])

        feed_dict = {gen.sids: sids, gen.class_labels: class_labels}
        feed_dict.update(train_data_dict)

        loss = {}
        class_scores, loss['dis'], loss['gen'], loss['class'], loss['overall'], _ = sess.run(
            [gen.real_class_scores, gen.dis_loss, gen.gen_loss, gen.class_loss, gen.loss, gen.opt], feed_dict)

        loss_list.append(loss)
        train_auc = utils.cal_auc(labels=class_labels, scores=class_scores)
        print('At step %i:\tdis_loss: %4.1f \tgen_loss: %4.1f \tclass_loss: %4.1f \ttrain_auc: %.1f'
            % (step, loss['dis'], loss['gen'], loss['class'], train_auc), flush=True)
        step += 1

    # -Test-
    test_class_scores = []
    test_sample_num = cf.sample_num['test']
    test_class_labels = cf.class_labels['test']
    for sample_idx in range(0, test_sample_num, cf.test_batch_size):
        sample_idx_end = np.clip(sample_idx + cf.test_batch_size, 0, test_sample_num)
        sids = np.arange(sample_idx, sample_idx_end)
        class_labels = test_class_labels[sample_idx: sample_idx_end]
        feed_dict = {gen.sids: sids, gen.class_labels: class_labels}
        feed_dict.update(test_data_dict)
        class_scores = sess.run(gen.real_class_scores, feed_dict)
        test_class_scores.extend(class_scores)
    auc = utils.cal_auc(test_class_labels, test_class_scores)
    if best_auc < auc:
        best_auc = auc
    auc_list.append(auc)

    if epoch == cf.epoch_num // 2:
        utils.write_results(auc_list, '%i_auc' & epoch)

    print('\nTest scores (%s):\nAUC (%%): %.1f (best %.1f)' % (cf.note, auc, best_auc))

utils.write_results(auc_list, 'auc')
sess.close()
