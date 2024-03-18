import torch
train_data = None
test_data = None
classification = None

def get_data(batch_size):
    r = torch.tensor([val for val in [0.0, 0.25, 0.5, 0.75, 1.0]]).cuda()
    r = torch.cartesian_prod(r, r)
    data = r.repeat(batch_size//25, 1)

    z = 0.5 * torch.ones_like(data[:, 0]).unsqueeze(1).cuda()
    data = torch.cat((data, z), dim=1)
    print(data.shape)
    return data


def get_classification(input):
    x = input[:, 0]
    y = input[:, 1]

    intervals = torch.arange(-0.35, 1.5, 0.25)
    length = len(intervals)
    lows = torch.tensor([intervals[i*2] for i in range(length//2)]).unsqueeze(0).repeat(x.shape[0], 1).T.cuda()
    ups = torch.tensor([intervals[i*2+1] for i in range(length//2)]).unsqueeze(0).repeat(x.shape[0], 1).T.cuda()

    x_in = torch.bitwise_and((lows <= x), (x <= ups))
    x_label = torch.bitwise_or(x_in[0], x_in[1])
    x_label = torch.bitwise_or(x_label, x_in[2])
    y_in = torch.bitwise_and((lows <= y), (y <= ups))
    y_label = torch.bitwise_or(y_in[0], y_in[1])
    y_label = torch.bitwise_or(y_label, y_in[2])

    res = ~torch.bitwise_xor(x_label, y_label)

    return res.float().cuda().unsqueeze(1)


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
    return train_data.clone().cuda()



