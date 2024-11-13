import numpy as np  

#导入npy文件路径位置
# test = np.load('G:\github_project\L2D\test3\off_policy\drlResult_30x8_30x8_Seed200.npy')
test = np.load('G:\github_project\L2D\test3_req_data_into_dataset\generatedData30_8_Seed200.npy')
# (100, 2, 30, 8)
# (2, 30, 8)
print(test[0])