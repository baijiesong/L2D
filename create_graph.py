import networkx as nx
import random
import numpy as np
# 创建一个空的图
G = nx.Graph()

# 添加n个节点
G.add_nodes_from(range(12))

# 随机连接节点，添加带权重的边
for node in G.nodes():
    num_edges = random.randint(4, 8)  # 每个节点连接4到8条边
    neighbors = random.sample([n for n in G.nodes() if n != node], num_edges)
    
    for neighbor in neighbors:
        # 为每条边设置一个随机权重
        weight = random.randint(1, 10)  # 权重范围是1到10
        G.add_edge(node, neighbor, weight=weight)

# 打印图的边及其权重
print("图的边和权重：")
for u, v, weight in G.edges(data='weight'):
    print(f"边 ({u}, {v}) 权重: {weight}")

# 可视化图（可选）
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)  # 布局
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
# 定义矩阵大小
num_jobs = len(G.nodes())  # 作业数量与节点数量一致
num_paths = len(G.edges())  # 路径数量为边的数量

# 初始化两个矩阵
processing_time_matrix = np.zeros((num_jobs, num_paths), dtype=int)
path_order_matrix = np.zeros((num_jobs, num_paths), dtype=int)
# 创建边索引映射（边编号）
edge_index_map = {edge: idx for idx, edge in enumerate(G.edges())}

# 计算每个作业的最短路径并填充矩阵
for job in range(num_jobs):
    source = job
    destination = (job + 1) % num_jobs  # 环状结构，最后一个作业指向第一个

    try:
        # 使用Dijkstra算法计算最短路径
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight')
        shortest_path_length = nx.shortest_path_length(G, source=source, target=destination, weight='weight')
        print(f"Job {source} -> Job {destination}: 最短路径为 {shortest_path}，总权重为 {shortest_path_length}")
        # 填充加工时间矩阵和路径顺序矩阵
        for i in range(len(shortest_path) - 1):
            edge = tuple(sorted((shortest_path[i], shortest_path[i + 1])))
            edge_idx = edge_index_map[edge]
            processing_time_matrix[job, edge_idx] = G[shortest_path[i]][shortest_path[i + 1]]['weight']

        # 将路径边的顺序填入矩阵
        edge_indices = [
            edge_index_map[tuple(sorted((shortest_path[i], shortest_path[i + 1])))]
            for i in range(len(shortest_path) - 1)
        ]
        path_order_matrix[job, :len(edge_indices)] = edge_indices

        # 随机补全路径顺序矩阵剩余部分
        remaining_indices = list(set(range(num_paths)) - set(edge_indices))
        random.shuffle(remaining_indices)
        path_order_matrix[job, len(edge_indices):] = remaining_indices[: (num_paths - len(edge_indices))]

    except nx.NetworkXNoPath:
        print(f"Job {source} -> Job {destination} 无法找到路径")
# except nx.NetworkXNoPath:
#          # 如果没有路径连接
#         print(f"Job {source} -> Job {destination}: False")

# 保存矩阵到 .npy 文件
output_path = "job_processing_matrices.npy"
np.save(output_path, {'processing_time_matrix': processing_time_matrix, 'path_order_matrix': path_order_matrix})

print(f"矩阵已保存至 {output_path}")
data = np.load("job_processing_matrices.npy", allow_pickle=True).item()
processing_time_matrix = data['processing_time_matrix']
path_order_matrix = data['path_order_matrix']

print("加工时间矩阵:")
print(processing_time_matrix)

print("\n路径顺序矩阵:")
print(path_order_matrix)
# # 寻找每个 job 的最短路径
# for job in range(12):
#     source = job
#     destination = (job + 1) % 12  # 目的节点是下一个节点，最后一个节点指向第一个节点
    
#     try:
#         # 使用 Dijkstra 算法计算最短路径和最短路径长度
#         shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight')
#         shortest_path_length = nx.shortest_path_length(G, source=source, target=destination, weight='weight')
#         print(f"Job {source} -> Job {destination}: 最短路径为 {shortest_path}，总权重为 {shortest_path_length}")
#     except nx.NetworkXNoPath:
#         # 如果没有路径连接
#         print(f"Job {source} -> Job {destination}: False")