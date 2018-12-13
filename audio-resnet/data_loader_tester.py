from custom_wav_loader import wavLoader
import torch

dataset = wavLoader('dataset/test')

test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=None, num_workers=4, pin_memory=True, sampler=None)

for k, (input, label) in enumerate(test_loader):
    print(input.size(), len(label))

## need to check wav file is correctly imported

import visdom
import numpy as np
import matplotlib.pyplot as plt
vis = visdom.Visdom(use_incoming_socket=False)

import librosa
y,sr = librosa.load('test.wav')
melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr=sr, n_mels=120,n_fft=1024))
# print(melgram[0])
vis.heatmap(melgram,opts=dict(title='test.wav'))
# vis.heatmap(melgram[0:20],opts=dict(title='test.wav'))

import librosa
y,sr = librosa.load('test_eq.wav')
melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr=sr, n_mels=120,n_fft=1024))
# melgram = melgram[0]# print(melgram[0])
vis.heatmap(melgram,opts=dict(title='test_eq.wav'))
# vis.heatmap(melgram[0:20],opts=dict(title='test_eq.wav'))


import librosa
y,sr = librosa.load('test_eq2.wav')
melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr=sr, n_mels=120,n_fft=1024))
vis.heatmap(melgram,opts=dict(title='test_eq2.wav'))
# vis.heatmap(melgram[0:20],opts=dict(title='test_eq2.wav'))

print(len(dataset[0][0][0]))
vis.heatmap(dataset[0][0][0])

len(dataset[0][0][0][0])
# vis.audio(dataset[0][0][0][0])

from custom_wav_loader import spect_loader
path='dataset/test/Lt OLV/2018:4:25:8:32:20 to 2018:4:25:8:33:26_Lt OLV000.wav'
window_size=0.02
window_stride=0.01
window_type='hamming'
window = True
normalize=True
a=spect_loader(path, window_size, window_stride, window, normalize, max_len=700)[0][0]


# spect_loader(path, window_size, window_stride, window, normalize, max_len=800)
plt.plot(spect_loader(path, window_size, window_stride, window, normalize, max_len=700).numpy()[0][0])

window_size=0.005   # 0.02
window_stride=0.01 # 0.01
window_type='hamming'
normalize=True
max_len=800

path='test.wav'
y, sr = librosa.load(path, sr=None)
n_fft = int(sr * window_size)
win_length = n_fft
hop_length = int(sr * window_stride)
print('sr',sr)
print('n_fft',n_fft)
print('win_length',win_length)
print('hop_length',hop_length)

# STFT

melgram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y, sr=sr, n_mels=120,n_fft=1024))

D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window_type)
spect, phase = librosa.magphase(D)

vis.heatmap(melgram,opts=dict(title='mel'))
# vis.heatmap(spect,opts=dict(title='1st'))

spect = np.log1p(spect)
# vis.heatmap(spect[0:20],opts=dict(title='after log'))



# make all spects with the same dims
# TODO: change that in the future
if spect.shape[1] < max_len:
    pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
    spect = np.hstack((spect, pad))
elif spect.shape[1] > max_len:
    spect = spect[:, :max_len]

# spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
# spect = torch.FloatTensor(spect)

# # z-score normalization
# if normalize:
#     mean = spect.mean()
#     std = spect.std()
#     if std != 0:
#         spect.add_(-mean)
#         spect.div_(std)

vis.heatmap(spect,opts=dict(title='hey'))

len(spect[0])

len(melgram[0])