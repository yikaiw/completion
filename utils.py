import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import config as cf
import embed
from reader import Reader


def get_data_dicts():
    reader = Reader()
    data_dict = {'train': {}, 'test': {}}
    id_embed_table_dict = {
        embed.id_embed_table['image']: reader.image_id_table, 
        embed.id_embed_table['video']: reader.video_id_table
    }
    for stage in ['train', 'test']:
        for modality in ['image', 'video']:
            data_dict[stage][embed.sample_target_ids[modality]] = reader.target_ids[stage][modality]
            data_dict[stage][embed.sample_history_ids[modality]] = reader.history_ids[stage][modality]
        data_dict[stage].update(id_embed_table_dict)
    return data_dict['train'], data_dict['test']


def get_negs(batch_size, neg_k):
    sample_num = cf.sample_num['train']['neg']
    idx = np.random.randint(0, sample_num, size=[batch_size * neg_k])
    neg_sids = cf.sample_idx[idx]
    return neg_sids


def cal_auc(labels, scores):
    auc = roc_auc_score(y_true=labels, y_score=scores) * 100
    return auc


def cal_pCTR(scores):
    pCTR = np.mean(scores) * 100
    return pCTR


def write_results(data_list, name, get_max=True):
    with open(os.path.join(cf.result_dir, name + '.txt'), 'a') as f:
        opt_score = data_list[0]
        f.write('%s|%s|' % (cf.note, cf.data_dates))
        for data in data_list:
            f.write('%.2f ' % data)
            if get_max ^ (opt_score > data):
                opt_score = data
        f.write('|%.3f\n' % opt_score)

