import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import config as cf
import embed
from reader import Reader


def get_data_dicts():
    reader = Reader()
    train_data_dict = {embed.id_embed_table['sku']: reader.id_embed_table['train']['sku'],
        embed.id_embed_table['video']: reader.id_embed_table['train']['video']}
    test_data_dict = {embed.id_embed_table['sku']: reader.id_embed_table['test']['sku'],
        embed.id_embed_table['video']: reader.id_embed_table['test']['video']}
    for modality_type in ['sku', 'video']:
        for sample_type in ['pos', 'neg']:
            train_data_dict[embed.sample_target_ids[modality_type][sample_type]] \
                = reader.sample_target_ids['train'][modality_type][sample_type]
            train_data_dict[embed.sample_history_ids[modality_type][sample_type]] \
                = reader.sample_history_ids['train'][modality_type][sample_type]
            test_data_dict[embed.sample_target_ids[modality_type][sample_type]] \
                = reader.sample_target_ids['test'][modality_type][sample_type]
            test_data_dict[embed.sample_history_ids[modality_type][sample_type]] \
                = reader.sample_history_ids['test'][modality_type][sample_type]
    return train_data_dict, test_data_dict


def dense_embedding(inputs, hidden_size, layer_num, name):
    state = inputs
    for layer in range(layer_num):
        layer_name = name + '_layer%i' % layer
        state = tf.layers.dense(state, hidden_size, tf.nn.relu, name=layer_name)
    return state


def get_negs(batch_size, cs):
    sample_num = cf.sample_num['train']['neg']
    idx = np.random.randint(0, sample_num, size=[batch_size * cf.neg_k])
    neg_sids = cf.sample_idx[idx]
    return neg_sids


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

