import numpy as np


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


def uni_instance_gen_res(n_j, n_m, low, high, missing_rate=0.5):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    # 在times矩阵中随机插入0值
    for i in range(n_j):
        zero_count = int(n_m * missing_rate)  # 每行的0值数量
        zero_indices = np.random.choice(n_m, zero_count, replace=False)
        times[i, zero_indices] = 0  # 将随机位置的值设置为0
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
   

    return times, machines

def override(fn):
    """
    override decorator
    """
    return fn


