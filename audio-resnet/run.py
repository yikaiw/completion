from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from custom_wav_loader import wavLoader
import numpy as np
from resnet import resnet34, resnet18
from train import train, test
import os
import visdom

vis = visdom.Visdom(use_incoming_socket=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_path = 'dataset/train/'
valid_path = 'dataset/valid/'
test_path = 'dataset/test/'

# parameter
optimizer = 'adadelta'  # adadelta adam SGD
lr = 0.001  # to do : adaptive lr
epochs = 1000
epoch = 1
momentum = 0.9  # for SGD

iteration = 0
patience = 5
log_interval = 100

seed = 1234  # random seed
batch_size = 20  # 100
test_batch_size = 10
arc = 'resnet18'

# sound setting
window_size = 0.01  # 0.02
window_stride = 0.01  # 0.01
window_type = 'hamming'
normalize = True

# loading data
train_dataset = wavLoader(
    train_path, window_size=window_size, window_stride=window_stride, window_type=window_type, normalize=normalize)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None)
valid_dataset = wavLoader(
    valid_path, window_size=window_size, window_stride=window_stride, window_type=window_type, normalize=normalize)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=None, num_workers=4, pin_memory=True, sampler=None)
test_dataset = wavLoader(
    test_path, window_size=window_size, window_stride=window_stride, window_type=window_type, normalize=normalize)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=None, num_workers=4, pin_memory=True, sampler=None)

model = resnet18()
print("ResNet")


model = torch.nn.DataParallel(model).cuda()

# define optimizer
if optimizer.lower() == 'adam':  # adadelta
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif optimizer.lower() == 'adadelta':  # adadelta
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
elif optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

best_valid_loss = np.inf

if os.path.isfile('./checkpoint/' + str(arc) + '.pth'):
    state = torch.load('./checkpoint/' + str(arc) + '.pth')
    print('load pre-trained model of ' + str(arc) + '\n')
    print(state)
    best_valid_loss = state['acc']

# visdom
loss_graph = vis.line(Y=np.column_stack([10, 10, 10]), X=np.column_stack([0, 0, 0]),
    opts=dict(title='loss', legend=['Train loss', 'Valid loss', 'Test loss'], showlegend=True, xlabel='epoch'))

# trainint with early stopping

print('\nStart training...')
while (epoch < epochs + 1) and (iteration < patience):
    train(train_loader, model, optimizer, epoch, True, log_interval)
    train_loss = test(train_loader, model, True, mode='Train loss')
    valid_loss = test(valid_loader, model, True, mode='Valid loss')
    test_loss = test(test_loader, model, True, mode='Test loss')

    if valid_loss > best_valid_loss:
        iteration += 1
        print('\nLoss was not improved, iteration {0}\n'.format(str(iteration)))
    else:
        print('\nSaving model of ' + str(arc) + '\n')
        iteration = 0
        best_valid_loss = valid_loss
        state = {'net': model.module if True else model, 'acc': valid_loss, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):  # model load should be
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + str(arc) + '.pth')

    vis.line(Y=np.column_stack([train_loss, valid_loss, test_loss]), X=np.column_stack([epoch, epoch, epoch]),
        win=loss_graph, update='append', opts=dict(legend=['Train loss', 'Valid loss', 'Test loss'], showlegend=True))
    epoch += 1
# test(test_loader,model,True,mode='test loss')
print('Finished!!')


from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

y_target =    [1, 1, 1, 0, 0, 2, 0, 3]
y_predicted = [1, 0, 1, 0, 0, 2, 1, 3]

cm = confusion_matrix(y_target=y_target, y_predicted=y_predicted, binary=False)

import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()


def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

