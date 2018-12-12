import os
import shutil
import argparse


def move_files(original_fold, data_fold, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_fold = os.path.join(data_fold, vals[0])
            os.makedirs(dest_fold, exist_ok=True)
            shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))


def create_train_fold(original_fold, data_fold, test_fold):
    # list dirs
    dir_names = list()
    for file in os.listdir(test_fold):
        if os.path.isdir(os.path.join(test_fold, file)):
            dir_names.append(file)

    # build train fold
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(test_fold, file)) and file in dir_names:
            shutil.move(os.path.join(original_fold, file), os.path.join(data_fold, file))


def make_dataset(original_fold, out_path):
    valid_path = os.path.join(original_fold, 'valid_list.txt')
    test_path = os.path.join(original_fold, 'test_list.txt')

    valid_fold = os.path.join(out_path, 'valid')
    test_fold = os.path.join(out_path, 'test')
    train_fold = os.path.join(out_path, 'train')

    os.makedirs(valid_fold, exist_ok=True)
    os.makedirs(test_fold, exist_ok=True)
    os.makedirs(train_fold, exist_ok=True)

    move_files(original_fold, valid_fold, valid_path)
    move_files(original_fold, test_fold, test_path)
    create_train_fold(original_fold, train_fold, test_fold)


original_fold = 'original/'
out_path = 'dataset/'

if __name__ == '__main__':
    make_dataset(original_fold, out_path)
    print('Done!!')