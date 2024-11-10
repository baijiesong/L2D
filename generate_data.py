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
    data = np.array([[[3, 3, 0, 0, 0, 0, 0, 0],
                      [3, 0, 3, 0, 0, 0, 0, 0],
                      [3, 0, 0, 3, 0, 0, 1, 1],
                      [3, 0, 0, 0, 3, 0, 1, 1],
                      [3, 0, 0, 0, 0, 3, 1, 1],
                      [0, 3, 3, 0, 0, 0, 0, 0],
                      [3, 3, 0, 0, 0, 0, 0, 0],
                      [0, 3, 0, 3, 0, 0, 1, 1],
                      [0, 3, 0, 0, 3, 0, 1, 1],
                      [0, 3, 0, 0, 0, 3, 1, 1],
                      [3, 0, 3, 0, 0, 0, 0, 0],
                      [0, 3, 3, 0, 0, 0, 0, 0],
                      [0, 0, 3, 3, 0, 0, 1, 1],
                      [0, 0, 3, 0, 3, 0, 1, 1],
                      [0, 0, 3, 0, 0, 3, 1, 1],
                      [0, 0, 0, 3, 3, 0, 0, 0],
                      [0, 0, 0, 3, 0, 3, 0, 0],
                      [3, 0, 0, 3, 0, 0, 1, 1],
                      [0, 3, 0, 3, 0, 0, 1, 1],
                      [0, 0, 3, 3, 0, 0, 1, 1],
                      [0, 0, 0, 3, 3, 0, 0, 0],
                      [0, 0, 0, 0, 3, 3, 0, 0],
                      [3, 0, 0, 0, 3, 0, 1, 1],
                      [0, 3, 0, 0, 3, 0, 1, 1],
                      [0, 0, 3, 0, 3, 0, 1, 1],
                      [0, 0, 0, 0, 3, 3, 0, 0],
                      [0, 0, 0, 3, 0, 3, 0, 0],
                      [3, 0, 0, 0, 0, 3, 1, 1],
                      [0, 3, 0, 0, 0, 3, 1, 1],
                      [0, 0, 3, 0, 0, 3, 1, 1]],

                     [[1, 2, 3, 4, 5, 6, 7, 8],
                      [1, 3, 2, 4, 5, 6, 7, 8],
                      [1, 7, 8, 4, 2, 3, 5, 6],
                      [1, 7, 8, 5, 2, 3, 4, 6],
                      [1, 7, 8, 6, 2, 3, 4, 5],
                      [2, 1, 3, 4, 5, 6, 7, 8],
                      [2, 3, 1, 4, 5, 6, 7, 8],
                      [2, 7, 8, 4, 1, 3, 5, 6],
                      [2, 7, 8, 5, 1, 3, 4, 6],
                      [2, 7, 8, 6, 1, 3, 4, 5],
                      [3, 1, 2, 4, 5, 6, 7, 8],
                      [3, 2, 1, 4, 5, 6, 7, 8],
                      [3, 7, 8, 4, 1, 2, 5, 6],
                      [3, 7, 8, 5, 1, 2, 4, 6],
                      [3, 7, 8, 6, 1, 2, 4, 5],
                      [4, 5, 1, 2, 3, 6, 7, 8],
                      [4, 6, 1, 2, 3, 5, 7, 8],
                      [4, 8, 7, 1, 2, 3, 5, 6],
                      [4, 8, 7, 2, 1, 3, 5, 6],
                      [4, 8, 7, 3, 1, 2, 5, 6],
                      [5, 4, 1, 2, 3, 6, 7, 8],
                      [5, 6, 1, 2, 3, 4, 7, 8],
                      [5, 8, 7, 1, 2, 3, 4, 6],
                      [5, 8, 7, 2, 1, 3, 4, 6],
                      [5, 8, 7, 3, 1, 2, 4, 6],
                      [6, 5, 1, 2, 3, 4, 7, 8],
                      [6, 4, 1, 2, 3, 5, 7, 8],
                      [6, 8, 7, 1, 2, 3, 4, 5],
                      [6, 8, 7, 2, 1, 3, 4, 5],
                      [6, 8, 7, 3, 1, 2, 4, 5]]])
    data =np.expand_dims(data, axis=0)
    print(data.shape)
    np.save('generatedData_acc_Seed.npy', data) 
    # 假设掩码矩阵，1表示需要加工，0表示不需要加工
    # mask = np.array([[1, 1, 1, 0, 0, 0],
    #                 [1, 1, 0, 1, 0, 0],
    #                 [1, 0, 0, 0, 1, 1],
    #                 [1, 1, 1, 0, 0, 0],
    #                 [0, 1, 1, 1, 0, 0],
    #                 [1, 0, 0, 1, 1, 1]])

    # # 修改加工时间矩阵
    # modified_times = np.where(mask == 1, data[0], 0)  # 使用np.inf表示不加工的机器

    # # 更新加工顺序矩阵
    # # 只保留需要加工的机器的索引
    # modified_order = []
    # for i in range(data[1].shape[0]):
    #     order = data[1][i][mask[i] == 1]  # 只保留需要加工的机器的顺序
    #     modified_order.append(order)

    # modified_order = np.array(modified_order, dtype=object)
    # print("Modified Processing Times:\n", modified_times)
    # print("Modified Processing Order:\n", modified_order)
    print(data)

def look(path):
    test = np.load(path)
    print(test)
if __name__ == '__main__':
    # t = gen_random_data()
    # look('G:\github_project\L2D\DataGen\generatedData30_8_Seed200.npy')
    gen_acc_data()