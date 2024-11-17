import networkx as nx
import random
import numpy as np
node=12
l_edges=4
h_edges=8
l_weight=1
h_weight=10
batch_size = 100
seed = 200
# 初始化边记录字典
edge_records = {}

# 边编号计数器
edge_id = 1

# 创建一个空的图
G = nx.Graph()

# 添加n个节点
G.add_nodes_from(range(node))

# 随机连接节点，添加带权重的边
for node in G.nodes():
    num_edges = random.randint(l_edges, h_edges)  # 每个节点连接4到8条边
    neighbors = random.sample([n for n in G.nodes() if n != node], num_edges)
    
    for neighbor in neighbors:
        if not G.has_edge(node, neighbor):  # 确保不会重复添加边
            # 为每条边设置一个随机权重
            weight = random.randint(l_weight, h_weight)  # 权重范围是1到10
            G.add_edge(node, neighbor, weight=weight)
            
            # 记录边编号和信息
            edge_records[edge_id] = {
                'source': node,
                'destination': neighbor,
                'weight': weight
            }
            
            # 打印边的编号和信息
            print(f"边编号 {edge_id}: {node} -> {neighbor}, 权重: {weight}")
            
            # 更新边编号
            edge_id += 1

# 打印所有边的记录
print("\n所有边记录：")
for eid, record in edge_records.items():
    print(f"边编号 {eid}: 来源 {record['source']}, 目的地 {record['destination']}, 权重 {record['weight']}")

# 检查图是否连通
if nx.is_connected(G):
    print("图是连通的")
else:
    print("图不是连通的")
    # 打印不连通的子图
    components = list(nx.connected_components(G))
    print(f"不连通的子图数量: {len(components)}")
    for i, component in enumerate(components):
        print(f"子图 {i + 1}: 节点 {component}")

# 可视化图（可选）
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)  # 布局
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()


# 构造路径记录
path_records = {}
# 寻找每个 job 的最短路径
for job in range(12):
    source = job
    destination = (job + 1) % 12  # 目的节点是下一个节点，最后一个节点指向第一个节点
    
    try:
        # 使用 Dijkstra 算法计算最短路径和最短路径长度
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight')
        shortest_path_length = nx.shortest_path_length(G, source=source, target=destination, weight='weight')
        print(f"Job {source} -> Job {destination}: 最短路径为 {shortest_path}，总权重为 {shortest_path_length}")
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
        # 如果没有路径连接
        path_records[job] = {
            'source': source,
            'destination': destination,
            'path': None,
            'length': float('inf'),
            'edges': []
        }
        print(f"Job {source} -> Job {destination}: False")
# 打印路径记录
print("\n路径记录：")
for job, record in path_records.items():
    print(f"Job {record['source']} -> {record['destination']}: 路径 {record['path']}, 权重 {record['length']}, 涉及边编号 {record['edges']}")
# 获取路径总数
num_paths = len(edge_records)
num_jobs = len(path_records)

# 初始化两个矩阵
processing_time_matrix = np.zeros((num_jobs, num_paths), dtype=int)  # 第一矩阵
path_order_matrix = np.zeros((num_jobs, num_paths), dtype=int)       # 第二矩阵

# 构造第一个矩阵
for job, record in path_records.items():
    if record['path']:
        for edge_id in record['edges']:
            # 将对应的路径编号的加工时间填入
            processing_time_matrix[job, edge_id - 1] = edge_records[edge_id]['weight']

# 构造第二个矩阵
all_path_ids = list(edge_records.keys())
for job, record in path_records.items():
    if record['path']:
        # 最短路径涉及的路径编号
        shortest_path_edges = record['edges']
    else:
        # 如果没有最短路径，用空列表
        shortest_path_edges = []
    
    # 补齐路径编号：先填最短路径涉及的编号，再随机补充其他路径编号
    remaining_paths = [pid for pid in all_path_ids if pid not in shortest_path_edges]
    random.shuffle(remaining_paths)  # 随机顺序
    complete_path_order = shortest_path_edges + remaining_paths
    
    # 填入矩阵，确保长度为 num_paths
    path_order_matrix[job, :] = complete_path_order[:num_paths]

# 打印结果
print("\n第一个矩阵（加工时间矩阵）：")
print(processing_time_matrix)
print("\n第二个矩阵（路径顺序矩阵）：")
print(path_order_matrix)
print(processing_time_matrix.shape)
print(path_order_matrix.shape)
