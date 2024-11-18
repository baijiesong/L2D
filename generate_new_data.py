import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt


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
            num_edges = random.randint(l_edges, h_edges)
            available_nodes = [i for i in G.nodes() if i != n]  # 获取除当前节点外的所有节点
            if num_edges > len(available_nodes):  # 如果随机生成的边数大于可供选择的节点总数
                num_edges = len(available_nodes)  # 将边数设置为可供选择的节点总数
            neighbors = random.sample(available_nodes, num_edges)
            
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
                    if G.number_of_edges() >= total_edges:  # 如果达到总边数，停止添加边
                        break
                # 如果遍历完节点，仍未达到总边数，则继续随机添加边
            if G.number_of_edges() >= total_edges:  # 如果达到总边数，停止添加边
                        break
        while G.number_of_edges() < total_edges:
            # 随机选择两个不同的节点
            node_a, node_b = random.sample(range(node), 2)
            
            # 确保边不存在，避免重复
            if not G.has_edge(node_a, node_b):
                weight = random.randint(l_weight, h_weight)
                G.add_edge(node_a, node_b, weight=weight)
                
                edge_records[edge_id] = {
                    'source': node_a,
                    'destination': node_b,
                    'weight': weight
                }
                edge_id += 1

            # 检查是否达到总边数
            if G.number_of_edges() == total_edges:
                break  
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(G)  # 布局
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig(f"./graph_image/generatedData{node*(node-1)}_{total_edges}_Seed{seed}/train/graph_{_ + 1}.png")  # 使用批次编号命名文件
        plt.close()  # 关闭当前绘图窗口，释放资源      

                # 构造路径记录
        path_records = {}
        job_id = 0  # 用于唯一标识每个 job
        for source in range(node):
            for destination in range(node):
                if source != destination:  # 确保源和目标不同
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

                        path_records[job_id] = {
                            'source': source,
                            'destination': destination,
                            'path': shortest_path,
                            'length': shortest_path_length,
                            'edges': shortest_path_edges
                        }
                    except nx.NetworkXNoPath:
                        path_records[job_id] = {
                            'source': source,
                            'destination': destination,
                            'path': None,
                            'length': float('inf'),
                            'edges': []
                        }

                    job_id += 1  # 更新 job_id

        # 获取路径总数和作业数
        num_jobs = len(path_records)
        num_paths = len(edge_records)
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
        print('succeed_instance!')
        

    # 保存结果到文件
    # ipdb.set_trace()
    np.save('./DataGen/generatedData{}_{}_Seed{}.npy'.format(job_id, total_edges, seed), batch)
    print(f"Batch of size {batch_size} saved as 'batch.npy'.")

if __name__ == '__main__':
    # 调用函数
    generate_batch(
        node=10,           # 节点数量
        l_edges=4,         # 每个节点的最少边数
        h_edges=8,         # 每个节点的最多边数
        l_weight=1,        # 边权重的最小值
        h_weight=10,       # 边权重的最大值
        batch_size=100,    # 批次大小
        seed=200,          # 随机种子
        total_edges=40     # 总边数
    )
    