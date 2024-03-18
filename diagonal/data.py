import torch
train_data = None
test_data = None


def get_data(batch_size):
    r = torch.tensor([(i/6*2-1) for i in range(7)]).unsqueeze(1).cuda()
    r = r.repeat(batch_size//7, 1)
    data = r.repeat(1, 2)
    return data


def get_classification(x):
    relevant = x[:, 0]
    intervals = torch.tensor([i / 6 * 2 - 0.9 for i in range(7)])
    length = len(intervals)
    lows = torch.tensor([intervals[i*2] for i in range(length//2)]).unsqueeze(0).repeat(relevant.shape[0], 1).T.cuda()
    ups = torch.tensor([intervals[i*2+1] for i in range(length//2)]).unsqueeze(0).repeat(relevant.shape[0], 1).T.cuda()
    return torch.any((lows <= relevant) & (relevant <= ups), dim=0).float().cuda().unsqueeze(1)


def get_data_iterator(data, batch_size):
    num_batches = data.shape[0] / batch_size
    i = 0
    while i < num_batches:
        data_p = data[i*batch_size: (i+1)*batch_size]
        target = get_classification(data_p)
        yield data_p, target
        i += 1


def init_data(data_size, test_data_size):
    global train_data, test_data
    train_data = get_data(data_size)
    test_data = get_data(test_data_size)


def get_test_iterator(batch_size):
    return get_data_iterator(test_data, batch_size=batch_size)


def get_train_iterator(batch_size):
    return get_data_iterator(train_data, batch_size=batch_size)


def get_train_data():
    global train_data
    return train_data.clone()
