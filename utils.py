import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import config as cf
import embed
from reader import Reader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean('restore', default=False, help='Restore from previous model and continue training, default False.')
tf.flags.DEFINE_boolean('baseline', default=False, help='Run uniform sampling as baseline, default False.')
tf.flags.DEFINE_string('note', default=None, help='Discription, default None is the current time.')
tf.flags.DEFINE_integer('gpu', default=0, help='Select gpu from 0 to 3, default 0.')
tf.flags.DEFINE_integer('cs', default=None, help='Select candidate size, default None.')
tf.flags.DEFINE_integer('neg_k', default=None, help='Select neg_k >= 1, default None.')
tf.flags.DEFINE_string('ns', default=None, help='Select ns_type from ons and ens, default None.')
tf.flags.DEFINE_string('d_loss', default=None, help='Select dis_loss from hinge and ce, default None.')
tf.flags.DEFINE_float('tem', default=None, help='Select temperature, default None.')
tf.flags.DEFINE_float('tem_d', default=None, help='Select temperature decay, default None.')

if FLAGS.cs is not None:
    cf.cs = FLAGS.cs
if FLAGS.ns is not None:
    cf.ns_ratio = 1. if FLAGS.ns == 'ons' else 0.
if FLAGS.d_loss is not None:
    cf.dis_loss = FLAGS.d_loss
    cf.ce_loss = True if cf.dis_loss == 'ce' else False
if FLAGS.neg_k is not None:
    cf.neg_k = FLAGS.neg_k
if cf.ns_ratio == 0.:
    cf.neg_k = 1
if FLAGS.i_lam is not None:
    cf.item_lambda = FLAGS.i_lam
if FLAGS.h_lam is not None:
    cf.history_lambda = FLAGS.h_lam
if FLAGS.tem is not None:
    cf.tem = FLAGS.tem
if FLAGS.tem_d is not None:
    cf.tem_decay = FLAGS.tem_d
if FLAGS.baseline is True:
    cf.cs, cf.gen_lr, cf.denoise, cf.epoch_start_gen = 1, 0., False, cf.epoch_num
ns_type = 'ons' if cf.ns_ratio == 1 else 'ens'
cf.note = '%s,cs=%i,k=%i,%s,lam=(%.1f,%.1f),t=%.1f,t_d=%.2f' \
    % (ns_type, cf.cs, cf.neg_k, cf.dis_loss, cf.item_lambda, cf.history_lambda, cf.tem, cf.tem_decay)
if FLAGS.note is not None:
    cf.note = FLAGS.note
    

def get_data_dicts():
    reader = Reader()
    train_data_dict = {embed.sample_item_ids['pos']: reader.item_ids['train']['pos'],
        embed.sample_item_ids['neg']: reader.item_ids['train']['neg'],
        embed.sample_history_ids['pos']: reader.history_ids['train']['pos'],
        embed.sample_history_ids['neg']: reader.history_ids['train']['neg'],
        embed.sample_embeds['pos']: reader.sample_embeds['train']['pos'],
        embed.sample_embeds['neg']: reader.sample_embeds['train']['neg'],
        embed.all_item_embeds: reader.all_item_embeds,
        embed.all_item_cids: reader.all_item_cids}
    test_data_dict = {embed.sample_item_ids['pos']: reader.item_ids['test']['pos'],
        embed.sample_item_ids['neg']: reader.item_ids['test']['neg'],
        embed.sample_history_ids['pos']: reader.history_ids['test']['pos'],
        embed.sample_history_ids['neg']: reader.history_ids['test']['neg'],
        embed.sample_embeds['pos']: reader.sample_embeds['test']['pos'],
        embed.sample_embeds['neg']: reader.sample_embeds['test']['neg'],
        embed.all_item_embeds: reader.all_item_embeds,
        embed.all_item_cids: reader.all_item_cids}
    item_ids, item_samples = reader.item_ids, reader.item_samples
    return train_data_dict, test_data_dict, item_ids, item_samples


def get_cands(batch_size, cs):
    sample_num = cf.sample_num['train']['neg']
    neg_cand_sids = np.random.randint(0, sample_num, size=[batch_size * cf.neg_k, cs])
    return neg_cand_sids


def get_denoised_cands(batch_size, cs, pos_sids, item_ids, item_samples, random=None):
    sample_num = cf.sample_num['train']['neg']
    # if cf.neg_sampling_method == 'random' or np.random.rand() < 0.1:
    if cf.neg_sampling_method == 'random' or random:
        neg_cand_sids = np.random.randint(0, sample_num, size=[batch_size, cs])
        for i in range(batch_size):
            for j in range(cs):
                while cf.neg_sid_noise[neg_cand_sids[i][j]]:
                    neg_cand_sids[i][j] = np.random.randint(0, sample_num)
        return neg_cand_sids

    neg_cand_sids = []
    for pos_sid in pos_sids:
        item_id = item_ids['train']['pos'][pos_sid]
        if item_id not in item_samples:
            neg_cand_sids.append(np.random.randint(0, sample_num, size=cs))
            continue
        item_samples_ = np.array(item_samples[item_id])
        item_sample_num = len(item_samples_)
        single_cand_sids = item_samples_.take(np.random.randint(0, item_sample_num, size=cs))
        for j in range(cs):
            while cf.neg_sid_noise[single_cand_sids[j]]:
                single_cand_sids[j] = item_samples_.take(np.random.randint(0, item_sample_num, size=1))
        neg_cand_sids.append(single_cand_sids)
    return neg_cand_sids


def sample_from_probs(neg_cand_probs):
    neg_cand_idxs = []
    for i in range(len(neg_cand_probs)):
        neg_cand_idx = np.random.choice(cf.cs, 1, p=neg_cand_probs[i])
        neg_cand_idxs.append([i, neg_cand_idx])
    return np.squeeze(neg_cand_idxs)


def get_neg_sids(neg_cand_sids, neg_cand_idxs):
    neg_sids = []
    for i in range(len(neg_cand_sids)):
        neg_sid = neg_cand_sids[i][neg_cand_idxs[i][1]]
        neg_sids.append(neg_sid)
    return np.squeeze(neg_sids)


def cal_auc(labels, scores):
    auc = roc_auc_score(y_true=labels, y_score=scores) * 100
    return auc


def cal_pCTR(scores):
    pCTR = np.mean(scores) * 100
    return pCTR


def write_results(data_list, name, decimal_2=True, get_max=True):
    with open(os.path.join(cf.result_dir, name + '.txt'), 'a') as f:
        opt_score = data_list[0]
        f.write('%s|%s|' % (cf.note, cf.data_dates))
        for data in data_list:
            f.write('%.2f ' % data) if decimal_2 else f.write('%.3f ' % data)
            if get_max and opt_score < data:
                opt_score = data
            elif not get_max and opt_score > data:
                opt_score = data
        f.write('|%.3f\n' % opt_score)

