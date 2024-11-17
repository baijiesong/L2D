import numpy as np  

#导入npy文件路径位置
test = np.load('G:\github_project\L2D\DataGen\generatedData30_8_Seed200.npy')
# test = np.load('G:\\github_project\\L2D\\test4_all_sim_req_data\\off_policy\\drlResult_30x8_30x8_Seed200.npy')
# (100, 2, 30, 8)
# (2, 30, 8)
print(test.shape)
# test[0], test[1] = test[1], test[0]
#np.save('./generatedData30_8_Seed200_modified.npy', test)
