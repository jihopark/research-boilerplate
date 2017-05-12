# codes related to batch generation

"""
def batch_gen(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size].T, y[i : (i+1)*batch_size].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield [x[i] for i in sample_idx], [y[i] for i in sample_idx]

# batch generator that gives out balanced batch for each class
def balanced_batch_gen(x, y, batch_size, balance=[0.5, 0.5]):
    classes = np.unique(y)

    # for now only works f:or binary classes with even number of batch_size
    assert len(classes) == 2 and batch_size % 2 == 0

    idx_1 = np.where( y == classes[0])[0]
    idx_2 = np.where( y == classes[1])[0]
    x_1 = [x[i] for i in idx_1]
    x_2 = [x[i] for i in idx_2]
    y_1 = [y[i] for i in idx_1]
    y_2 = [y[i] for i in idx_2]

    print("Generating batch of %s with distribution of %.2f %.2f" %
            (batch_size, balance[0], balance[1]))

    while True:
        sample_idx_1 = sample(list(np.arange(len(x_1))),
                int(batch_size*balance[0]))
        sample_idx_2 = sample(list(np.arange(len(x_2))),
                int(batch_size*balance[1]))
        yield (np.concatenate(
                ([x_1[i] for i in sample_idx_1],
                 [x_2[i] for i in sample_idx_2])),
               np.concatenate(
                ([y_1[i] for i in sample_idx_1],
                 [y_2[i] for i in sample_idx_2])))
"""
