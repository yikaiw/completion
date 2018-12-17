import os
import urllib2
import numpy as np
import config as cf


class Reader(object):
    def __init__(self):
        self.build_paths()
        self.index_load_sku()
        self.index_load_video()
        self.load_samples()

    def build_paths(self):
        def path_dict(filename):
            train_paths, test_paths = [], []
            for dir in cf.sample_dirs['train']:
                train_paths.append(os.path.join(cf.data_dir, dir, filename))
            for dir in cf.sample_dirs['test']:
                test_paths.append(os.path.join(cf.data_dir, dir, filename))
            return {'train': train_paths, 'test': test_paths}

        def join(filename):
            return os.path.join(cf.data_dir, filename)

        self.stages = ['train', 'test']
        self.sample_file = path_dict('user_clicked_videos.dat')
        self.sku_embed_file = join('embed_sku.dat')
        self.video_embed_file = [join('embed_video_1.dat'), join('embed_video_2.dat')]

    def index_load_sku(self):
        # index sku ids and load sku embeddings
        self.sku_id, self.sku_id_num = {}, 1
        self.sku_embed = []
        print('Re-index ids and load sku embeddings.', flush=True)
        sku_embed_file = open(self.sku_embed_file, 'r')
        for id_embed in sku_embed_file.readlines():
            id = id_embed.split()[0]
            if id not in self.sku_id:
                try:
                    embed = list(map(float, id_embed.split()[1:]))
                    self.sku_embed.append(embed)
                except ValueError:
                    continue
                self.sku_id[id] = self.sku_id_num
                self.sku_id_num += 1
        sku_embed_file.close()
        print('sku id num %i\n' % self.sku_id_num, flush=True)
        cf.sku_id_num = self.sku_id_num

    def index_load_video(self):
        # index video ids and load sku embeddings
        self.video_id, self.video_id_num = {}, 1
        self.video_embed = []
        print('Re-index ids and load video embeddings.', flush=True)
        for i in range(len(self.video_embed_file)):
            video_embed_file = open(self.video_embed_file[i], 'r')
            for id_embed in video_embed_file.readlines():
                id = id_embed.split()[0]
                if id not in self.video_id:
                    try:
                        embed = list(map(float, id_embed.split()[1:]))
                        self.video_embed.append(embed)
                    except ValueError:
                        continue
                    self.video_id[id] = self.video_id_num
                    self.video_id_num += 1
            video_embed_file.close()
        print('video id num %i\n' % self.video_id_num, flush=True)
        cf.video_id_num = self.video_id_num

    def load_samples(self):
        def get_list_ids(origin_ids, id_table):
            ids = []
            for origin_id in origin_ids:
                if origin_id in id_table:
                    ids.append(id_table[origin_id])
                else:
                    ids.append(0)  # 0 stands for missing
            return ids

        self.target_ids = {'sku': {'train': [], 'test': []}, 'video': {'train': [], 'test': []}}
        self.history_ids = {'sku': {'train': [], 'test': []}, 'video': {'train': [], 'test': []}}
        self.label = {'train': [], 'test': []}
        for stage in self.stages:
            print('Load samples in [%s] data.' % stage), flush=True)
            total_sample_num, used_sample_num = {'pos': 0, 'neg': 0}, {'pos': 0, 'neg': 0}

            for idx in range(len(cf.sample_dirs[stage])):
                sample_file = open(self.sample_file[stage][idx], 'r')
                for sample in sample_file.readlines():
                    split = sample.split(',')
                    sample_type = 'pos' if split[0] == '1' else 'neg'
                    request_id, request_time, user_id, image_id, video_id, sku_id = split[1: 7]
                    total_sample_num[sample_type] += 1
                    if sample.count('mp4') <= 1 or video_id not in self.video_id or sku_id not in self.sku_id:
                        continue
                    used_sample_num[sample_type] += 1
                    history_sku_ids = get_list_ids(split[8].split(' '), self.sku_id)
                    video_origin_ids = [i.split('_')[0] for i in split[7].split(' ')]
                    history_video_ids = get_list_ids(video_origin_ids, self.video_id)
                    # history_video_times = [i.split('_')[1] for i in history_video]
                    self.label[stage].append(int(sample_type == 'pos'))
                    self.target_ids['sku'][stage].append(sku_id)
                    self.target_ids['video'][stage].append(video_id)
                    self.history_ids['sku'][stage].append(history_sku_ids)
                    self.history_ids['video'][stage].append(history_video_ids)
                sample_file.close()
            print('total sample num in [%s] data: pos %i, neg %i\n' 
                % (stage, total_sample_num['pos'], total_sample_num['neg']), flush=True)
            print('used sample num in [%s] data: pos %i, neg %i\n' 
                % (stage, used_sample_num['pos'], used_sample_num['neg']), flush=True)

    def url_loader(self):
        url, image_id, opener = '', '', ''
        url = 'http://storage.jd.com' + url
        videoname = os.path.basename(url)
        os.mkdir(image_id)
        videopath = os.path.join(image_id, videoname)
        proxy_handler = urllib2.ProxyHandler({'http': 'http://172.22.178.101:80'})
        opener = urllib2.build_opener(opener)
        with open(videopath, 'wb') as f:
            f.write(urllib2.urlopen(url).read())