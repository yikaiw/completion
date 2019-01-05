import os
import tqdm
import numpy as np
import argparse
import warnings
import config as cf
warnings.filterwarnings('ignore')

def join(filename):
    return os.path.join(cf.data_dir, filename)


class Filtrate(object):
    def __init__(self):
        self.build_paths()
        self.load_samples()
        self.filtrate_embed_image()
        self.filtrate_embed_video()

    def build_paths(self):
        self.stages = ['train', 'test']
        self.sample_file = join('user_clicked_videos.dat')
        self.image_embed_file = join('embed_sku.dat')
        self.video_embed_file = [join('embed_video_1.dat'), join('embed_video_2.dat')]
        self.image_embed_output_file = join('_embed_image.dat')
        self.video_embed_output_file = join('_embed_video.dat')

    def load_samples(self):
        print('Load samples.', flush=True)
        self.origin_id_dict = {'image': {}, 'video': {}}
        sample_file = open(self.sample_file, 'r')
        for sample in tqdm.tqdm(sample_file.readlines()):
            split = sample.split(',')
            request_id, request_time, user_id, image_id, video_id, image_id = split[1: 7]
            if sample.count('mp4') <= 1:
                continue
            target_id, history_id = {}, {}
            target_id['image'], target_id['video'] = image_id, video_id
            image_origin_ids = [i.split('_')[1] for i in split[7].split(' ')]  # sku ids
            for image_origin_id in image_origin_ids:
                self.origin_id_dict['image'][image_origin_id] = True
            video_origin_ids = [i.split('_')[0] for i in split[7].split(' ')]
            for video_origin_id in video_origin_ids:
                self.origin_id_dict['video'][video_origin_id] = True
        sample_file.close()

    def filtrate_embed_image(self):
        print('Filtrate image embeddings.', flush=True)
        assert not os.path.exists(self.image_embed_output_file)
        image_embed_file = open(self.image_embed_file, 'r')
        image_embed_output_file = open(self.image_embed_output_file, 'w')
        image_embed = image_embed_file.readline()
        total_num, valid_num = 0, 0
        while image_embed:
            total_num += 1
            id = image_embed.split()[0]
            if id in self.origin_id_dict['image'] and self.origin_id_dict['image'][id]:
                self.origin_id_dict['image'][id] = False
                image_embed_output_file.write(image_embed)
                valid_num += 1
            image_embed = image_embed_file.readline()
        image_embed_file.close()
        image_embed_output_file.close()
        print('total num: %i, valid num: %i' % (total_num, valid_num), flush=True)

    def filtrate_embed_video(self):
        print('Filtrate video embeddings.', flush=True)
        assert not os.path.exists(self.video_embed_output_file)
        video_embed_output_file = open(self.video_embed_output_file, 'w')
        total_num, valid_num = 0, 0
        for i in range(len(self.video_embed_file)):
            video_embed_file = open(self.video_embed_file[i], 'r')
            video_embed = video_embed_file.readline()
            while video_embed:
                total_num += 1
                id = video_embed.split()[0]
                if id in self.origin_id_dict['video'] and self.origin_id_dict['video'][id]:
                    self.origin_id_dict['video'][id] = False
                    video_embed_output_file.write(video_embed)
                    valid_num += 1
                video_embed = video_embed_file.readline()
            video_embed_file.close()
        video_embed_output_file.close()
        print('total num: %i, valid num: %i' % (total_num, valid_num), flush=True)


def split_samples():
    print('spliting user_clicked_videos.dat into into train and test, train_rate %.2f' % cf.train_rate)
    assert not os.path.exists(join('_samples_train.dat'))
    sample_file = open(join('user_clicked_videos.dat'), 'r')
    sample_file_train = open(join('_samples_train.dat'), 'w')
    sample_file_test = open(join('_samples_test.dat'), 'w')
    for line in tqdm.tqdm(sample_file.readlines()):
        if np.random.rand() < cf.train_rate:
            sample_file_train.write(line)
        else:
            sample_file_test.write(line)
    sample_file.close()
    sample_file_train.close()
    sample_file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtrate', action='store_true', default=False, help='Filtrate embeddings')
    parser.add_argument('--split_sample', action='store_true', default=False, help='Split samples')
    args = parser.parse_args()

    if args.filtrate:
        filtrate = Filtrate()
    if args.split_sample:
        split_samples()
