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


def revise():
    data = np.load('G:\github_project\L2D\DataGen\generatedData30_8_Seed200.npy', allow_pickle=True)

    # Step 2: 添加新的数组元素
    new_element = np.array([[[3, 3, 0, 0, 0, 0, 0, 0],
                      [0, 3, 3, 0, 0, 0, 0, 0],
                      [3, 3, 0, 0, 0, 0, 0, 0],
                      [0, 3, 0, 3, 0, 0, 1, 1],
                      [0, 3, 0, 0, 3, 0, 1, 1],
                      [0, 3, 0, 0, 0, 3, 1, 1],
                      [3, 0, 3, 0, 0, 0, 0, 0],
                      [3, 0, 0, 3, 0, 0, 1, 1],
                      [3, 0, 0, 0, 3, 0, 1, 1],
                      [3, 0, 0, 0, 0, 3, 1, 1],
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
                      [6, 8, 7, 3, 1, 2, 4, 5]]])  # 这是要添加的元素
    data = np.append(data, [new_element], axis=0)  # 确保 axis 设置正确
    data = np.delete(data, 0, axis=0)
    print(data.shape)
    # Step 3: 保存更新后的数组到原文件或新文件
    np.save('G:\github_project\L2D\DataGen\generatedData30_8_Seed200.npy', data)


def own_data():
    base_data= np.array([[[3, 3, 0, 0, 0, 0, 0, 0],
                      [0, 3, 3, 0, 0, 0, 0, 0],
                      [3, 3, 0, 0, 0, 0, 0, 0],
                      [0, 3, 0, 3, 0, 0, 1, 1],
                      [0, 3, 0, 0, 3, 0, 1, 1],
                      [0, 3, 0, 0, 0, 3, 1, 1],
                      [3, 0, 3, 0, 0, 0, 0, 0],
                      [3, 0, 0, 3, 0, 0, 1, 1],
                      [3, 0, 0, 0, 3, 0, 1, 1],
                      [3, 0, 0, 0, 0, 3, 1, 1],
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
    # 生成100个样本
    samples = []
    samples.append(base_data)
    for _ in range(99):
        # 随机打乱第一个矩阵中的每一行内容
        first_matrix_shuffled_rows = np.array([np.random.permutation(row) for row in base_data[0]])
        # 随机打乱第一个矩阵的行顺序
        first_matrix_shuffled = first_matrix_shuffled_rows[np.random.permutation(first_matrix_shuffled_rows.shape[0])]

        # 随机打乱第二个矩阵中的每一行内容
        second_matrix_shuffled_rows = np.array([np.random.permutation(row) for row in base_data[1]])
        # 随机打乱第二个矩阵的行顺序
        second_matrix_shuffled = second_matrix_shuffled_rows[np.random.permutation(second_matrix_shuffled_rows.shape[0])]

        # 将生成的矩阵组合为一个样本
        samples.append([first_matrix_shuffled, second_matrix_shuffled])
    np.save('G:\github_project\L2D\DataGen\generatedData30_8_Seed200.npy', samples)
    vali_samples=[]
    for _ in range(100):
        # 随机打乱第一个矩阵中的每一行内容
        first_matrix_shuffled_rows = np.array([np.random.permutation(row) for row in base_data[0]])
        # 随机打乱第一个矩阵的行顺序
        first_matrix_shuffled = first_matrix_shuffled_rows[np.random.permutation(first_matrix_shuffled_rows.shape[0])]

        # 随机打乱第二个矩阵中的每一行内容
        second_matrix_shuffled_rows = np.array([np.random.permutation(row) for row in base_data[1]])
        # 随机打乱第二个矩阵的行顺序
        second_matrix_shuffled = second_matrix_shuffled_rows[np.random.permutation(second_matrix_shuffled_rows.shape[0])]

        # 将生成的矩阵组合为一个样本
        samples.append([first_matrix_shuffled, second_matrix_shuffled])
    np.save('G:\github_project\L2D\DataGen\Vali\generatedData30_8_Seed200.npy', vali_samples)
     
if __name__ == '__main__':
    # t = gen_random_data()
    # gen_random_data()
    # revise()
    own_data()
    look('G:\github_project\L2D\DataGen\generatedData30_8_Seed200.npy')