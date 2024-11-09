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
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0).astype(float) 
    machines = permute_rows(machines)
    
    # 在每一行中添加缺失值
    for i in range(n_j):
        missing_count = int(n_m * missing_rate)  # 每行的缺失数量
        missing_indices = np.random.choice(n_m, missing_count, replace=False)
        machines[i, missing_indices] = np.nan  # 用 np.nan 表示缺失值

    return times, machines

def override(fn):
    """
    override decorator
    """
    return fn


