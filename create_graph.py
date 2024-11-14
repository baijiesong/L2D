import networkx as nx
import random

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
# 寻找每个 job 的最短路径
for job in range(12):
    source = job
    destination = (job + 1) % 12  # 目的节点是下一个节点，最后一个节点指向第一个节点
    
    try:
        # 使用 Dijkstra 算法计算最短路径和最短路径长度
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight')
        shortest_path_length = nx.shortest_path_length(G, source=source, target=destination, weight='weight')
        print(f"Job {source} -> Job {destination}: 最短路径为 {shortest_path}，总权重为 {shortest_path_length}")
    except nx.NetworkXNoPath:
        # 如果没有路径连接
        print(f"Job {source} -> Job {destination}: False")