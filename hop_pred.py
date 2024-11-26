import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import heapq
import random
import matplotlib.pyplot as plt
import time


# 定义画图函数
def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

# 跳数计算函数
def calculate_hops(graph):
    """计算每对节点之间的跳数（Shortest Path Length）。"""
    G = to_networkx(graph, to_undirected=True)
    num_nodes = graph.num_nodes
    hop_matrix = torch.full((num_nodes, num_nodes), float('inf'), dtype=torch.float)  # 初始化为无穷大

    for source_node in range(num_nodes):
        lengths = nx.single_source_shortest_path_length(G, source=source_node)
        for target_node, hop_count in lengths.items():
            hop_matrix[source_node, target_node] = hop_count

    return hop_matrix


# 节点特征生成函数
def generate_features(graph, time_step):
    G = to_networkx(graph, to_undirected=True)
    degrees = torch.tensor([G.degree(node) for node in range(graph.num_nodes)], dtype=torch.float)
    clustering_coeffs = torch.tensor([nx.clustering(G, node) for node in range(graph.num_nodes)], dtype=torch.float)
    time_features = torch.full((graph.num_nodes,), time_step, dtype=torch.float)  # 添加时间片特征

    # 将度数、聚类系数和时间片特征堆叠到一起
    x = torch.stack([degrees, clustering_coeffs, time_features], dim=1)
    return x


# 图的创建
def process_data_from_file(file_path):
    # 读取数据
    data = pd.read_csv(file_path, header=None)
    data.columns = ['时间片', '起始点', '终止点', '距离']
    data['起始点'], start_labels = pd.factorize(data['起始点'])
    data['终止点'], end_labels = pd.factorize(data['终止点'])
    num_nodes = len(start_labels)

    # 初始化变量
    graphs = []

    # 根据时间片分组
    for time_step, group in tqdm(data.groupby('时间片')):
        # 构造 edge_index 和 edge_weight
        edge_index = torch.tensor(group[['起始点', '终止点']].values.T, dtype=torch.long)
        scaler = MinMaxScaler()
        edge_weight = torch.tensor(scaler.fit_transform(group['距离'].values.reshape(-1, 1)), dtype=torch.float).squeeze()

        # 在这里生成节点特征
        temp_graph = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
        x = generate_features(temp_graph, time_step)  # 生成节点特征

        # 创建图数据并直接传递节点特征
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)

        # 计算跳数标签
        hops = calculate_hops(graph)
        graph.hop_labels = hops  # 将整个跳数矩阵保存到图数据对象中
        graphs.append(graph)

    # 使用 80% 作为训练集，20% 作为测试集
    split_idx = int(0.8 * len(graphs))
    train_dataset = graphs[:split_idx]
    test_dataset = graphs[split_idx:]

    return train_dataset, test_dataset, num_nodes


class GNNRoutingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=4):
        super(GNNRoutingModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.high_order_conv = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.fc1 = nn.Linear(hidden_dim * heads * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)  # 30%的dropout概率


    def forward(self, data):
        # 提取稀疏图数据
        x, edge_index = data.x, data.edge_index

        # 低阶特征提取
        low_order_features = F.leaky_relu(self.conv1(x, edge_index))
        low_order_features = F.leaky_relu(self.conv2(low_order_features, edge_index))

        # 高阶特征提取
        high_order_features = F.leaky_relu(self.high_order_conv(low_order_features, edge_index))

        # 构建所有节点对的特征组合矩阵
        num_nodes = x.size(0)
        row_idx, col_idx = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
        row_idx, col_idx = row_idx.flatten(), col_idx.flatten()
        x_pairwise = torch.cat([high_order_features[row_idx], high_order_features[col_idx]], dim=1)

        # 使用全连接层预测跳数
        x_out = self.dropout(F.relu(self.fc1(x_pairwise)))  # [num_nodes * num_nodes, hidden_dim]
        x_out = self.fc2(x_out).view(num_nodes, num_nodes)  # [num_nodes, num_nodes]

        return x_out


def train_model():
    losses = []  # 用于存储每个 epoch 的损失值

    class Args:
        input_size = 3
        hidden_size = 256
        output_size = 1
        epochs = 1
        learning_rate = 0.001  # 调整学习率
        weight_decay = 1e-3  # 添加权重衰减防止过拟合

    args = Args()

    # 构建时间序列数据集
    train_loader, test_loader, num_nodes = process_data_from_file('data/trimmed_data_all.csv')
    model = GNNRoutingModel(input_dim=args.input_size,
                            hidden_dim=args.hidden_size)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # 每10个epoch学习率衰减
    criterion_hop = nn.MSELoss()  # 使用均方误差损失
    for epoch in range(args.epochs):
        # print(f"Epoch {epoch + 1}, num_nodes: {num_nodes}")
        model.train()
        epoch_hop_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            hop_pred = model(data)  # [num_nodes, num_nodes]
            hop_targets = data.hop_labels.float()  # 获取跳数标签矩阵 [num_nodes, num_nodes]

            loss_hop = criterion_hop(hop_pred, hop_targets)
            loss_hop.backward()
            optimizer.step()
            epoch_hop_loss += loss_hop.item()

        losses.append(epoch_hop_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, hop_Loss: {epoch_hop_loss / len(train_loader)}')
        # 学习率调度器更新
        scheduler.step()

    # 绘制损失值变化图
    plot_loss(losses)


    # 在测试集上进行评估
    test(model, test_loader)


# 测试函数定义
def test(model, test_loader):
    model.eval()
    all_hop_preds = []
    all_hop_labels = []
    overall_accuracy = []

    with torch.no_grad():
        for data in test_loader:
            hop_pred = model(data)
            # 对每个图分别提取相应的标签
            hop_targets = data.hop_labels.float()  # 获取跳数标签矩阵 [num_nodes, num_nodes]

            all_hop_preds.append(hop_pred.cpu().numpy())
            all_hop_labels.append(hop_targets.cpu().numpy())

            # 对每个样本调用 compare_routing_accuracy
            accuracy = compare_routing_accuracy(model, data)  # 比较 GNN 和 Dijkstra
            overall_accuracy.append(accuracy)

    # 对于每个时间步，计算 AUC
    hop_preds_flat = np.concatenate([pred.flatten() for pred in all_hop_preds], axis=0)  # 展平所有预测
    hop_labels_flat = np.concatenate([label.flatten() for label in all_hop_labels], axis=0)  # 展平所有标签

    mse_hop = mean_squared_error(hop_labels_flat, hop_preds_flat)

    print(f'Test hop mean: {mse_hop}')

    # 平均准确率
    avg_accuracy = np.mean(overall_accuracy) * 100
    print(f'Test routing accuracy (average): {avg_accuracy:.2f}%')

    # compare_routing_accuracy(model, data)

# 使用 Dijkstra 算法进行所有节点到所有节点的路径预测
def dijkstra_all_pairs(data):
    num_nodes = data.x.size(0)

    edge_index = data.edge_index  # 当前时间步的边索引，形状为 [2, 80]
    edge_attr = data.edge_attr  # 当前时间步的边属性（权重），形状为 [80]

    # 创建邻接表以表示图
    adj_list = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        start_node = edge_index[0, i].item()
        end_node = edge_index[1, i].item()
        weight = edge_attr[i].item()
        adj_list[start_node].append((end_node, weight))

    # 使用 Dijkstra 计算所有节点对的最短路径跳数
    all_hops = torch.full((num_nodes, num_nodes), float('inf'), dtype=torch.float)
    for start_node in range(num_nodes):
        visited = [False] * num_nodes
        min_heap = [(0, start_node)]  # (距离, 节点)
        all_hops[start_node, start_node] = 0

        while min_heap:
            current_hop, current_node = heapq.heappop(min_heap)
            if visited[current_node]:
                continue
            visited[current_node] = True

            for neighbor, _ in adj_list[current_node]:
                if not visited[neighbor]:
                    new_hop = current_hop + 1
                    if new_hop < all_hops[start_node, neighbor]:
                        all_hops[start_node, neighbor] = new_hop
                        heapq.heappush(min_heap, (new_hop, neighbor))

    return all_hops


# 比较 GNN 和 Dijkstra 的最短路径预测
def predict_all_pairs_routing_with_cache(model, data):
    model.eval()
    path_cache = {}
    with torch.no_grad():
        start_time_model = time.time()
        hop_predictions = model(data).cpu()
        hop_matrix = hop_predictions.squeeze()  # 尝试去掉不必要的维度
        end_time_model = time.time()
        print(f'神经网络加载所需时间: {end_time_model - start_time_model} 秒')

        start_time_for = time.time()
        # 遍历所有目标节点和源节点
        num_nodes = hop_matrix.size(0)
        for target_node in range(num_nodes):
            for current_node in range(num_nodes):
                # 如果缓存中已经有该路径，跳过计算
                if (current_node, target_node) in path_cache:
                    continue
                # 路径初始化
                path = [current_node]
                visited = [False] * num_nodes  # 创建一个 visited 数组，跟踪已访问节点
                visited[current_node] = True  # 当前节点标记为已访问

                while current_node != target_node :
                    neighbors = data.edge_index[1][data.edge_index[0] == current_node]
                    # 如果当前节点直接与目标节点相连
                    if target_node in neighbors:
                        path.append(target_node)
                        path_cache[(current_node, target_node)] = path
                        break

                    # 计算与目标节点的跳数，选择跳数最小的邻居节点
                    hop_values = hop_matrix[target_node, neighbors]

                    # 排除已经在路径中的节点
                    valid_neighbors = [neighbor for neighbor in neighbors if not visited[neighbor]]
                    valid_hop_values = hop_values[torch.tensor([neighbor in valid_neighbors for neighbor in neighbors])]

                    if valid_neighbors:
                        next_hop = valid_neighbors[torch.argmin(valid_hop_values)].item()
                        path.append(next_hop)
                        visited[next_hop] = True  # 标记该节点已访问
                        current_node = next_hop
                    else:
                        break  # 如果没有有效的邻居，退出循环

                # 缓存路径中所有节点的计算路径和跳数
                for i in range(len(path)):
                    if (path[i], target_node) not in path_cache:
                        path_cache[(path[i], target_node)] = path[i:]

        end_time_for = time.time()
        print(f'bb计算所需时间: {end_time_for - start_time_for} 秒')
    return path_cache



# 比较 GNN 和 Dijkstra 的最短路径预测
def compare_routing_accuracy(model, data):
    # GNN 模型路径预测
    start_time_gnn = time.time()
    gnn_paths = predict_all_pairs_routing_with_cache(model, data)
    end_time_gnn = time.time()
    print(f'GNN 所有节点到所有节点路径预测时间: {end_time_gnn - start_time_gnn} 秒')

    # Dijkstra 路径预测
    start_time_dijkstra = time.time()
    dijkstra_paths = dijkstra_all_pairs(data)
    end_time_dijkstra = time.time()
    print(f'Dijkstra 所有节点到所有节点路径预测时间: {end_time_dijkstra - start_time_dijkstra} 秒')

    # 比较路径预测的准确率
    correct_count = 0
    total_count = 0
    num_nodes = data.x.size(0)

    for start_node in range(num_nodes):
        for target_node in range(num_nodes):
            dijkstra_hop_count = dijkstra_paths[start_node, target_node].item()
            if dijkstra_hop_count == float('inf'):
                continue  # 忽略无路径的节点对

            gnn_path = gnn_paths.get((start_node, target_node), None)
            if gnn_path is not None:
                # 判断 GNN 预测的路径长度是否与 Dijkstra 一致
                if len(gnn_path) - 1 == dijkstra_hop_count:
                    correct_count += 1
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f'GNN 路径预测的准确度: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    train_model()



