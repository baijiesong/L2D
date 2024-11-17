import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import ipdb

def generate_batch(node, l_edges, h_edges, l_weight, h_weight, batch_size, seed,total_edges):
    """
    批量生成加工时间矩阵和路径顺序矩阵。

    Args:
        node (int): 节点数量。
        l_edges (int): 每个节点的最少边数。
        h_edges (int): 每个节点的最多边数。
        l_weight (int): 边权重的最小值。
        h_weight (int): 边权重的最大值。
        batch_size (int): 批次大小。
        seed (int): 随机种子。

    Returns:
        None: 结果保存在 `batch.npy` 文件中。
    """
    random.seed(seed)  # 设置随机种子
    np.random.seed(seed)
    
    batch = []  # 用于存储 batch_size 个元素

    for _ in range(batch_size):
        # 初始化边记录字典
        edge_records = {}
        edge_id = 1

        # 创建一个空的图
        G = nx.Graph()

        # 添加n个节点
        G.add_nodes_from(range(node))

        # 随机连接节点，添加带权重的边
        for n in G.nodes():
            while G.number_of_edges() < total_edges:
                num_edges = random.randint(l_edges, h_edges)
                neighbors = random.sample([i for i in G.nodes() if i != n], num_edges)
                
                for neighbor in neighbors:
                    if not G.has_edge(n, neighbor):  # 确保不会重复添加边
                        weight = random.randint(l_weight, h_weight)
                        G.add_edge(n, neighbor, weight=weight)
                        
                        edge_records[edge_id] = {
                            'source': n,
                            'destination': neighbor,
                            'weight': weight
                        }
                        edge_id += 1
                        if G.number_of_edges() == total_edges:  # 如果达到总边数，停止添加边
                            break
        if G.number_of_edges() == total_edges:  # 如果达到总边数，停止循环
            break

        # 构造路径记录
        path_records = {}
        for job in range(node):
            source = job
            destination = (job + 1) % node  # 目的节点是下一个节点
            
            try:
                # 使用 Dijkstra 算法计算最短路径和最短路径长度
                shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight')
                shortest_path_length = nx.shortest_path_length(G, source=source, target=destination, weight='weight')

                # 找到路径中涉及的边编号
                shortest_path_edges = [
                    eid for i in range(len(shortest_path) - 1)
                    for eid, record in edge_records.items()
                    if {record['source'], record['destination']} == {shortest_path[i], shortest_path[i+1]}
                ]
                
                path_records[job] = {
                    'source': source,
                    'destination': destination,
                    'path': shortest_path,
                    'length': shortest_path_length,
                    'edges': shortest_path_edges
                }
            except nx.NetworkXNoPath:
                path_records[job] = {
                    'source': source,
                    'destination': destination,
                    'path': None,
                    'length': float('inf'),
                    'edges': []
                }

        # 获取路径总数和作业数
        num_paths = len(edge_records)
        num_jobs = len(path_records)

        # 初始化两个矩阵
        processing_time_matrix = np.zeros((num_jobs, num_paths), dtype=int)
        path_order_matrix = np.zeros((num_jobs, num_paths), dtype=int)

        # 构造第一个矩阵
        for job, record in path_records.items():
            if record['path']:
                for edge_id in record['edges']:
                    processing_time_matrix[job, edge_id - 1] = edge_records[edge_id]['weight']

        # 构造第二个矩阵
        all_path_ids = list(edge_records.keys())
        for job, record in path_records.items():
            if record['path']:
                shortest_path_edges = record['edges']
            else:
                shortest_path_edges = []
            
            remaining_paths = [pid for pid in all_path_ids if pid not in shortest_path_edges]
            random.shuffle(remaining_paths)
            complete_path_order = shortest_path_edges + remaining_paths
            
            path_order_matrix[job, :] = complete_path_order[:num_paths]

        # 添加到 batch 中
        batch.append([processing_time_matrix, path_order_matrix])
        

    # 保存结果到文件
    # ipdb.set_trace()
    np.save("batch.npy", batch)
    print(f"Batch of size {batch_size} saved as 'batch.npy'.")

if __name__ == '__main__':
    # 调用函数
    generate_batch(
        node=16,           # 节点数量
        l_edges=4,         # 每个节点的最少边数
        h_edges=8,         # 每个节点的最多边数
        l_weight=1,        # 边权重的最小值
        h_weight=10,       # 边权重的最大值
        batch_size=100,    # 批次大小
        seed=200,          # 随机种子
        total_edges=56     # 总边数
    )
    test=np.load("batch.npy")
    print(test.shape)
    print(test)