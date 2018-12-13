import csv


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        return [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]


def get_std(norm_value=255):
    # Kinetics (10 videos for each class)
    return [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]
