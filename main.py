import numpy as np
import tensorflow as tf
import os
import warnings
from model import GAN
import utils
import config as cf

args = cf.parse_args()
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
print('Building GAN.')
gan = GAN()

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
    sess.run(gan.lr_update, feed_dict={gan.new_lr: lr})
    for sample_idx in range(0, train_sample_num, cf.batch_size):
        pos_sids = permu[sample_idx: sample_idx + cf.batch_size]
        neg_sids = utils.get_negs(cf.batch_size, cf.neg_k)  # [batch_size * neg_k]

        feed_dict = {dis.pos_sids: pos_sids, dis.neg_sids: neg_sids}
        feed_dict.update(train_data_dict)
        scores, rewards, dis_loss, _ = sess.run([dis.scores, dis.rewards, dis.loss, dis.opt], feed_dict)
        sess.run(gen_.lr_update, feed_dict={gen_.new_lr: gen_lr})
        feed_dict = {gen_.neg_cand_sids: neg_cand_sids, gen_.neg_cand_idxs: neg_cand_idxs, gen_.tem: tem,
            gen_.pos_sids: pos_sids, gen_.rewards: rewards, gen_.baseline: baseline}
        feed_dict.update(train_data_dict)
        gen_loss, _ = sess.run([gen_.loss, gen_.opt], feed_dict)

        dis_loss_list.append(dis_loss)
        labels = np.concatenate([[1] * len(scores['pos']), [0] * len(scores['neg'])])
        train_auc = utils.cal_auc(labels=labels, scores=np.concatenate([scores['pos'], scores['neg']]))
        gen_loss_list.append(gen_loss)

        print('At step %i (%s):\tdis_loss: %4.1f  \tgen_loss: %6.1f  \ttrain_auc: %.1f'
                % (step, ns_type, dis_loss, gen_loss + 1e-3, train_auc), flush=True)
        step += 1
        baseline = np.mean(rewards)
    # baseline = np.mean(rewards_list)

    # -Test-
    scores, loss = [], []
    for sample_type in ['pos', 'neg']:
        test_sample_num = cf.sample_num['test'][sample_type]
        if cf.test_all_data or epoch == cf.epoch_num - 1:
            data_num = test_sample_num
            permu = np.arange(data_num)
        else:
            data_num = cf.test_subsample_num[sample_type]
            permu = np.random.randint(0, test_sample_num, size=data_num)
        for sample_idx in range(0, data_num, cf.test_batch_size):
            sample_idx_end = np.clip(sample_idx + cf.test_batch_size, 0, data_num)
            batch_lines = permu[sample_idx: sample_idx_end]
            if sample_type == 'pos':
                feed_dict = {dis.pos_sids: batch_lines}
            else:
                feed_dict = {dis.neg_item_sids: batch_lines, dis.neg_history_sids: batch_lines}
            feed_dict.update(test_data_dict)
            if cf.ce_loss:
                batch_labels = test_labels[sample_idx: sample_idx_end]
                feed_dict.update({dis.labels: batch_labels})
                batch_scores, batch_loss = sess.run([dis.scores[sample_type], dis.test_loss[sample_type]], feed_dict)
                loss.append(batch_loss)
            else:
                batch_scores = sess.run(dis.scores[sample_type], feed_dict)
            scores.extend(batch_scores)
    auc = utils.cal_auc(test_labels, scores)
    if best_auc < auc:
        best_auc = auc
    auc_list.append(auc)

    if epoch == 26:
        utils.write_results(auc_list, '25_auc')

    print('\nTest scores (%s %s):\nAUC (%%): %.1f (best %.1f)' % (cf.note, str(cf.data_dates), auc, best_auc))

utils.write_results(auc_list, 'auc')
if cf.ce_loss:
    utils.write_results(pCTR_list, 'pCTR', True, False)
    utils.write_results(loss_list, 'loss', False, False)

sess.close()
