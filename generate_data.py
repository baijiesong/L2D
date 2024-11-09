import numpy as np
from uniform_instance_gen import uni_instance_gen,uni_instance_gen_res

j = 30
m = 8
l = 1
h = 4
batch_size = 100
seed = 200
def gen_random_data():
    np.random.seed(seed)
    # 先 jobs 再 machines
    data = np.array([uni_instance_gen_res(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    print(data.shape)
    np.save('generatedData{}_{}_Seed{}.npy'.format(j, m, seed), data) 
    path = 'generatedData{}_{}_Seed{}.npy'.format(j, m, seed)
    return path
def gen_acc_data():
    data = np.array([[[27, 17, 69, 43, 56, 77],
                      [80, 90, 15, 92, 58, 90],
                      [12,  7, 43, 57,  2,  8],
                      [52, 28, 74, 16,  2, 24],
                      [86, 25,  8, 23, 55, 78],
                      [17, 71, 36, 32, 33, 46]],

                     [[6,  4,  1,  3,  5,  2],
                      [3,  2,  1,  6,  5,  4],
                      [2,  5,  1,  6,  3,  4],
                      [2,  1,  3,  5,  4,  6],
                      [1,  4,  3,  5,  6,  2],
                      [2,  3,  6,  5,  4,  1]]])
    # 假设掩码矩阵，1表示需要加工，0表示不需要加工
    mask = np.array([[1, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [1, 0, 0, 1, 1, 1]])

    # 修改加工时间矩阵
    modified_times = np.where(mask == 1, data[0], 0)  # 使用np.inf表示不加工的机器

    # 更新加工顺序矩阵
    # 只保留需要加工的机器的索引
    modified_order = []
    for i in range(data[1].shape[0]):
        order = data[1][i][mask[i] == 1]  # 只保留需要加工的机器的顺序
        modified_order.append(order)

    modified_order = np.array(modified_order, dtype=object)
    print("Modified Processing Times:\n", modified_times)
    print("Modified Processing Order:\n", modified_order)
    print(data)

def look(path):
    test = np.load(path)
    print(test)
if __name__ == '__main__':
    t = gen_random_data()
    look(t)
    # gen_acc_data()