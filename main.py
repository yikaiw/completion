import numpy as np
import tensorflow as tf
import os
from os.path import join
import warnings
from generator import Generator
from discriminator import Discriminator
import utils
import config as cf
warnings.filterwarnings('ignore')
tf.set_random_seed(cf.random_seed)
np.random.seed(cf.random_seed)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

FLAGS = tf.flags.FLAGS
# os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

print('Task name: %s' % cf.note)
print('Loading data.')
train_data_dict, test_data_dict, item_ids, item_samples = utils.get_data_dicts()
print('Building generator.')
gen = {'ons': Generator('ons'), 'ens': Generator('ens')}
print('Building discriminator.')
dis = Discriminator()

auc_ = tf.placeholder(tf.float32)
auc_summary_op = tf.summary.scalar('auc', auc_)

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
dis_writer = tf.summary.FileWriter(join(cf.log_dir, 'dis@' + cf.note), sess.graph)
gen_writer = tf.summary.FileWriter(join(cf.log_dir, 'gen@' + cf.note), sess.graph)
test_writer = tf.summary.FileWriter(join(cf.log_dir, 'test@' + cf.note), sess.graph)

model_path = join(cf.model_dir, 'model-')
if not FLAGS.restore:
    sess.run(tf.global_variables_initializer())
    step = 0
else:
    print('Restoring from pre-trained model.')
    saver.restore(sess, model_path)
    step = int(str(os.listdir(cf.model_dir)[-1].split('-')[-1]).split('.')[0])

baseline, auc, tem, dis_lr = 0., 0., cf.tem, cf.dis_lr
neg_sids, dis_loss_list, gen_loss_list = [], [], []
auc_list, pCTR_list, f_score_list, g_score_list, loss_list = [], [], [], [], []
train_sample_num = cf.sample_num['train']['pos'] // cf.batch_size * cf.batch_size
test_labels = np.concatenate([np.ones(cf.sample_num['test']['pos']), np.zeros(cf.sample_num['test']['neg'])])
best_auc = 0.

print('Start training:')
for epoch in range(cf.epoch_num):
    # -Train-
    print('\nFor epoch %i (/%i):' % (epoch, cf.epoch_num - 1))
    permu = np.random.permutation(train_sample_num)
    rewards_list = []
    dis_lr = cf.dis_lr * cf.lr_decay ** (epoch // cf.lr_decay_epoch)
    sess.run(dis.lr_update, feed_dict={dis.new_lr: dis_lr})
    tem = cf.tem * cf.tem_decay ** epoch
    gen_lr = cf.gen_lr * cf.lr_decay ** (epoch // cf.lr_decay_epoch)
    for sample_idx in range(0, train_sample_num, cf.batch_size):
        pos_sids = permu[sample_idx: sample_idx + cf.batch_size]
        neg_cand_sids = utils.get_cands(cf.batch_size, cf.cs)  # [batch_size * neg_k, cs]

        ns_type = 'ons' if np.random.rand() < cf.ns_ratio else 'ens'
        gen_ = gen[ns_type]
        feed_dict = {gen_.pos_sids: pos_sids, gen_.neg_cand_sids: neg_cand_sids, gen_.tem: tem}
        feed_dict.update(train_data_dict)
        # neg_cand_probs = np.ones((cf.batch_size, cf.cs)) / cf.cs
        neg_cand_probs = sess.run(gen_.neg_cand_probs, feed_dict)  # [batch_size, cs]
        # print(neg_cand_probs[0], flush=True)
        neg_cand_idxs = utils.sample_from_probs(neg_cand_probs)  # [batch_size, 2]
        neg_sids = utils.get_neg_sids(neg_cand_sids, neg_cand_idxs)  # [batch_size]
        dis_neg_his_sids = neg_sids if cf.ns_type == 'ons' else pos_sids

        feed_dict = {dis.pos_sids: pos_sids, dis.neg_item_sids: neg_sids, dis.neg_history_sids: dis_neg_his_sids}
        feed_dict.update(train_data_dict)
        scores, rewards, dis_loss, _, dis_summary = sess.run(
            [dis.scores, dis.rewards, dis.loss, dis.opt, dis.summary_op], feed_dict)
        sess.run(gen_.lr_update, feed_dict={gen_.new_lr: gen_lr})
        feed_dict = {gen_.neg_cand_sids: neg_cand_sids, gen_.neg_cand_idxs: neg_cand_idxs, gen_.tem: tem,
            gen_.pos_sids: pos_sids, gen_.rewards: rewards, gen_.baseline: baseline}
        feed_dict.update(train_data_dict)
        gen_loss, _, gen_summary = sess.run([gen_.loss, gen_.opt, gen_.summary_op], feed_dict)

        dis_loss_list.append(dis_loss)
        dis_writer.add_summary(dis_summary, step)
        labels = np.concatenate([[1] * len(scores['pos']), [0] * len(scores['neg'])])
        train_auc = utils.cal_auc(labels=labels, scores=np.concatenate([scores['pos'], scores['neg']]))
        gen_loss_list.append(gen_loss)
        gen_writer.add_summary(gen_summary, step)

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

    if auc > 70.5 and cf.test_all_data and cf.save_model:
        save_path = saver.save(sess, '%s(%d)' % (model_path + cf.note, int(auc)), global_step=step)
        print('Model saved in file: %s' % save_path)
    auc_summary = sess.run(auc_summary_op, {auc_: auc})
    test_writer.add_summary(auc_summary, epoch)

utils.write_results(auc_list, 'auc')
if cf.ce_loss:
    utils.write_results(pCTR_list, 'pCTR', True, False)
    utils.write_results(loss_list, 'loss', False, False)

sess.close()
dis_writer.close()
gen_writer.close()
