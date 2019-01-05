import os
import numpy as np
import argparse
import warnings
import config as cf
warnings.filterwarnings('ignore')

def join(filename):
    return os.path.join(cf.data_dir, filename)


class Reader(object):
    def __init__(self):
        self.build_paths()
        self.index_load_embed_image()
        self.index_load_embed_video()
        self.load_samples()

    def build_paths(self):
        self.sample_file = {'train': join('_samples_train.dat'), 'test': join('_samples_test.dat')}
        self.image_embed_file = join('_embed_image.dat')
        self.video_embed_file = join('_embed_video.dat')

    def index_load_embeds_image(self):
        # index image ids and load image embeddings
        self.image_id_table, self.image_id_num = {}, 1
        self.image_embed = []
        print('Re-index ids and load image embeddings.', flush=True)
        image_embed_file = open(self.image_embed_file, 'r')
        id_embed = image_embed_file.readline()
        while id_embed:
            id = id_embed.split()[0]
            if id not in self.image_id_table:
                try:
                    embed = list(map(float, id_embed.split()[1:]))
                    self.image_embed.append(embed)
                except ValueError:
                    continue
                self.image_id_table[id] = self.image_id_num
                self.image_id_num += 1
            id_embed = image_embed_file.readline()
        image_embed_file.close()
        self.image_id_table[0] = 0
        print('image id num %i\n' % self.image_id_num, flush=True)
        cf.image_id_num = self.image_id_num

    def index_load_embeds_video(self):
        # index video ids and load image embeddings
        self.video_id_table, self.video_id_num = {}, 1
        self.video_embed = []
        print('Re-index ids and load video embeddings.', flush=True)
        for i in range(len(self.video_embed_file)):
            video_embed_file = open(self.video_embed_file[i], 'r')
            id_embed = video_embed_file.readline()
            while id_embed:
                id = id_embed.split()[0]
                if id not in self.video_id_table:
                    try:
                        embed = list(map(float, id_embed.split()[1:]))
                        self.video_embed.append(embed)
                    except ValueError:
                        continue
                    self.video_id_table[id] = self.video_id_num
                    self.video_id_num += 1
                id_embed = video_embed_file.readline()
            video_embed_file.close()
        self.video_id_table[0] = 0
        print('video id num %i\n' % self.video_id_num, flush=True)
        cf.video_id_num = self.video_id_num

    def load_samples(self):
        def get_list_ids(origin_ids, id_table):
            origin_ids_tmp = np.zeros(cf.history_len)
            len_min = min(cf.history_len, len(origin_ids))
            origin_ids_tmp[:len_min] = origin_ids[:len_min]
            ids = []
            for origin_id in origin_ids_tmp:
                ids.append(id_table[origin_id] if origin_id in id_table else 0)
            return ids

        self.target_ids = {'image': {'train': [], 'test': []}, 'video': {'train': [], 'test': []}}
        self.history_ids = {'image': {'train': [], 'test': []}, 'video': {'train': [], 'test': []}}
        self.class_labels = {'train': [], 'test': []}
        self.sample_idx = {'train': {'pos': [], 'neg': []}, 'test': {'pos': [], 'neg': []}}
        for stage in ['train', 'test']:
            print('Load samples in [%s] data.' % stage, flush=True)
            total_sample_num, used_sample_num = {'pos': 0, 'neg': 0}, {'pos': 0, 'neg': 0}

            sample_file = open(self.sample_file[stage], 'r')
            for sample in sample_file.readlines():
                split = sample.split(',')
                sample_type = 'pos' if split[0] == '1' else 'neg'
                request_id, request_time, user_id, image_id, video_id, image_id = split[1: 7]
                total_sample_num[sample_type] += 1
                if sample.count('mp4') <= 1:
                    continue
                if video_id not in self.video_id_table or image_id not in self.image_id_table:
                    continue
                target_id, history_id = {}, {}
                target_id['image'], target_id['video'] = image_id, video_id
                image_origin_ids = [i.split('_')[1] for i in split[7].split(' ')]  # sku ids
                history_id['image'] = get_list_ids(image_origin_ids, self.image_id_table)
                video_origin_ids = [i.split('_')[0] for i in split[7].split(' ')]
                history_id['video'] = get_list_ids(video_origin_ids, self.video_id_table)
                # history_video_times = [i.split('_')[1] for i in history_video]
                if stage == 'test':
                    for i in range(cf.history_len):
                        if np.random.rand < cf.missing_rate:
                            history_id['video'][i] = 0
                self.class_labels[stage].append(int(sample_type == 'pos'))
                self.sample_idx[stage][sample_type].append(used_sample_num[sample_type])
                for modality in ['image', 'video']:
                    self.target_ids[modality][stage].append(target_id[modality])
                    self.history_ids[modality][stage].append(history_id[modality])
                used_sample_num[sample_type] += 1
            sample_file.close()
            print('total sample num in [%s] data: pos %i, neg %i\n' 
                % (stage, total_sample_num['pos'], total_sample_num['neg']), flush=True)
            print('used sample num in [%s] data: pos %i, neg %i\n' 
                % (stage, used_sample_num['pos'], used_sample_num['neg']), flush=True)
            cf.sample_num[stage] = used_sample_num['pos'] + used_sample_num['neg']
        cf.class_labels, cf.sample_idx = self.class_labels, self.sample_idx


if __name__ == '__main__':
    print('testing class Reader()')
    reader = Reader()
