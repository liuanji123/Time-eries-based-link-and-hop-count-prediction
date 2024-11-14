
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
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
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 检查是否有可用的GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# 节点特征生成函数
def generate_features(graph):
    G = to_networkx(graph, to_undirected=True)
    degrees = torch.tensor([G.degree(node) for node in range(graph.num_nodes)], dtype=torch.float)
    clustering_coeffs = torch.tensor([nx.clustering(G, node) for node in range(graph.num_nodes)], dtype=torch.float)
    x = torch.stack([degrees, clustering_coeffs], dim=1)
    return x

# 图的创建
def process_data_from_file(file_path, seq_len, batch_size):
    # 读取数据
    data = pd.read_csv(file_path, header=None)
    data.columns = ['时间片', '起始点', '终止点', '距离']
    data['起始点'], start_labels = pd.factorize(data['起始点'])
    data['终止点'], end_labels = pd.factorize(data['终止点'])
    num_nodes = len(start_labels)
    print(f"num_nodes: {num_nodes}")
    # 初始化变量
    graphs = []

    # 根据时间片分组
    for time_step, group in tqdm(data.groupby('时间片')):
        # 构造 edge_index 和 edge_weight
        edge_index = torch.tensor(group[['起始点', '终止点']].values.T, dtype=torch.long)
        scaler = MinMaxScaler()
        edge_weight = torch.tensor(scaler.fit_transform(group['距离'].values.reshape(-1, 1)), dtype=torch.float).squeeze()

        # 创建图数据并生成节点特征
        graph = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
        graph.x = generate_features(graph)
        graphs.append(graph)

    # 生成时间序列样本
    samples = []
    for i in range(len(graphs) - seq_len):
        # 获取每一个时间步的图数据
        seq_graphs = graphs[i:i + seq_len]

        # 这里保留每个时间步图的节点特征
        x = torch.stack([g.x for g in seq_graphs], dim=0)  # (seq_len, num_nodes, feature_dim)

        # 每个时间步的边信息都保留独立，注意我们直接保留每个时间步的 edge_index 和 edge_attr
        edge_indices = [g.edge_index for g in seq_graphs]  # (seq_len, num_edges, 2)
        print(edge_indices[0].shape) # list(2 * 80) 8个元素
        edge_weights = [g.edge_attr for g in seq_graphs]  # (seq_len, num_edges)

        samples.append((x, edge_indices, edge_weights)) # (1440 - 8) * ()

    # 划分训练集和测试集
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

    # 创建 DataLoader
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_nodes

# 定义GCN模型用于时间序列链接预测和跳数预测
class GNNRoutingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes):
        super(GNNRoutingModel, self).__init__()
        self.num_nodes = num_nodes  # 将 num_nodes 存储为类的属性
        # 两个卷积层用于提取拓扑特征
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 链路预测的全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 结合高阶和低阶特征
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 链路预测输出
        # 跳数预测的全连接层
        self.fc_hop = nn.Linear(hidden_dim * 2, num_nodes)  # 调整输出维度

    def forward(self, data):
        # print("x shape",data.x.shape)
        seq_len = data.x.size(0)  # 获取序列长度
        # batch_size = data.x.size(1)  # 获取批量大小
        num_nodes = self.num_nodes  # 获取节点数

        x = data.x  # (seq_len, num_nodes, feature_dim)
        edge_indices = data.edge_index  # (seq_len, num_edges, 2)
        edge_weights = data.edge_attr  # (seq_len, num_edges)

        # print(f"x.shape: {x.shape}")  # 应该是 (seq_len, num_nodes, feature_dim)

        # 初始化输出
        link_out = []
        hop_out = []

        # 对每个时间步的图分别进行卷积操作
        for t in range(seq_len):
            x_t = x[t]  # 将 x[t] 调整为二维张量 (num_nodes, feature_dim)
            edge_index = edge_indices[t]
            edge_attr = edge_weights[t]

            # 图卷积操作
            x_low = F.relu(self.conv1(x_t, edge_index, edge_attr))
            x_low = F.relu(self.conv2(x_low, edge_index, edge_attr))

            x_high = F.relu(self.conv1(x_t, edge_index, edge_attr))
            x_high = F.relu(self.conv2(x_high, edge_index, edge_attr))

            # 特征交叉
            x_combined = torch.cat([x_low, x_high], dim=-1)  # 通过拼接交叉特征

            # 链路预测
            link_pred = F.relu(self.fc1(x_combined))
            link_out.append(self.fc2(link_pred))  # 每个时间步的链路预测输出

            # 跳数预测
            hop_out.append(self.fc_hop(x_combined))  # 每个时间步的跳数预测输出
        # # 聚合每个时间步的预测输出（例如，取平均）
        link_out = torch.stack(link_out, dim=0)  # (seq_len, num_nodes, num_nodes)
        hop_out = torch.stack(hop_out, dim=0)  # (seq_len, num_nodes, num_nodes)

        return link_out, hop_out


# 定义 Dijkstra 函数以计算最短路径和跳数
def dijkstra_shortest_hops(graph, source_node):
    G = to_networkx(graph, to_undirected=True)
    lengths = nx.single_source_shortest_path_length(G, source=source_node)
    return lengths


# # 计算跳数的函数
# def calculate_hops(graphs, seq_len):
#     hop_targets = []

#     for t in range(seq_len):  # seq_len 时间步
#         graph_t = graphs[t]  # 选择第 t 个时间步的图数据
#         print(f"graph_t.num_nodes: {graph_t.num_nodes}")

#         # 对每个时间步，计算每个节点对的跳数
#         for source_node in range(graph_t.num_nodes):
#             lengths = dijkstra_shortest_hops(graph_t, source_node)
#             for target_node in range(graph_t.num_nodes):
#                 hop_targets.append(lengths.get(target_node, float('inf')))

#     return torch.tensor(hop_targets, dtype=torch.float)
def calculate_hops(graphs, seq_len):
    # 初始化三维张量来存储每个时间步的跳数 (seq_len, num_nodes, num_nodes)
    hop_targets = []

    for t in range(seq_len):  # seq_len 时间步
        graph_t = graphs[t]  # 选择第 t 个时间步的图数据

        # 创建一个零矩阵来保存该时间步所有节点对的跳数
        hop_matrix = torch.zeros((graph_t.num_nodes, graph_t.num_nodes), dtype=torch.float)

        # 对每个时间步，计算每个节点对的跳数
        for source_node in range(graph_t.num_nodes):
            lengths = dijkstra_shortest_hops(graph_t, source_node)
            for target_node in range(graph_t.num_nodes):
                hop_matrix[source_node, target_node] = lengths.get(target_node, float('inf'))

        hop_targets.append(hop_matrix)  # 将当前时间步的跳数矩阵添加到列表中

    # 将列表中的跳数矩阵堆叠成一个三维张量
    return torch.stack(hop_targets, dim=0)  # 返回形状为 (seq_len, num_nodes, num_nodes) 的三维张量



# 计算负量采样
def negative_sample(data, neg_ratio=3):
    num_pos_samples = data.edge_index.size(1)
    num_neg_samples = num_pos_samples * neg_ratio

    # 如果 edge_index 是三维的, 遍历每个时间步
    edge_label_index_list = []
    for t in range(data.edge_index.size(0)):  # 如果数据包含多个时间步
        edge_index = data.edge_index[t]  # 选择第 t 个时间步的边

        # 确保 edge_index 是二维的
        if edge_index.dim() == 1:
            # 如果是 1D 的，手动调整为 2D
            edge_index = edge_index.unsqueeze(0)  # 在第一维加一维，使其成为 2D
            edge_index = edge_index.repeat(2, 1)  # 重复两次使其符合 edge_index 的形状

        assert edge_index.dim() == 2, f"edge_index should be 2D, but got {edge_index.dim()}D"

        neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=num_neg_samples, method='sparse'
        )
        edge_label_index_list.append(torch.cat([edge_index, neg_edge_index], dim=-1).long())

    edge_label_index = torch.cat(edge_label_index_list, dim=-1)
    edge_label = torch.cat([torch.ones(num_pos_samples), torch.zeros(num_neg_samples)], dim=0)
    perm = torch.randperm(edge_label.size(0))
    edge_label = edge_label[perm]
    edge_label_index = edge_label_index[:, perm]
    return edge_label, edge_label_index


def train_model():
    losses = []  # 用于存储每个 epoch 的损失值

    class Args:
        seq_len = 8
        input_size = 2
        hidden_size = 256
        output_size = 1
        batch_size = 32

    args = Args()

    # 构建时间序列数据集
    train_loader, test_loader, num_nodes = process_data_from_file('data/trimmed_data_all.csv', args.seq_len,
                                                                  args.batch_size)
    print(f"Number of nodes: {num_nodes}")
    model = GNNRoutingModel(input_dim=args.input_size,
                            hidden_dim=args.hidden_size,
                            output_dim=args.output_size,
                            num_nodes=num_nodes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion_link = nn.MSELoss()
    criterion_hop = nn.MSELoss()  # 跳数预测的损失函数
    print(f"Model initialized with num_nodes: {model.num_nodes}")
    for epoch in range(2):
        print(f"Epoch {epoch + 1}, num_nodes: {num_nodes}")
        model.train()
        epoch_loss = 0

        for x, edge_indices, edge_weights in train_loader:
            optimizer.zero_grad()

            batch_size = x.shape[0]
            seq_len = x.shape[1]  # 序列长度
            num_nodes = x.shape[2]  # 节点数量
            feature_dim = x.shape[3]  # 特征维度


            losses_per_batch = []
            for i in range(batch_size):
                hop_targets = []  # 每个批次开始时初始化 hop_targets
                graphs = []  # 用于保存每个时间步的图数据
                edge_index_tot = []
                edge_weight_tot = []
                x_tot = []
                for t in range(seq_len):  # seq_len
                    edge_index = edge_indices[t][i]  # 获取第 t 个时间步的边
                    # print(f"edge_index shape at time step {t}: {edge_index.shape}")
                    data = Data(x=x[i, t].view(-1, x.size(-1)).to(device),  # 获取第t个时间步的节点特征
                                edge_index=edge_indices[t][i].to(device),  # 获取第t个时间步的边
                                edge_attr=edge_weights[t][i].to(device))  # 获取第t个时间步的边权重
                    edge_index_tot.append(edge_indices[t][i])
                    edge_weight_tot.append(edge_weights[t][i])
                    x_tot.append(x[i, t].view(-1, x.size(-1)))
                    graphs.append(data)  # 将每个时间步的图数据保存到 graphs 中
               # 将列表转换为张量
                x_tot = torch.stack(x_tot, dim=0)  # 将所有时间步的节点特征堆叠为一个张量，形状为 (seq_len, num_nodes, feature_dim)
                edge_index_tot = torch.stack(edge_index_tot, dim=0)  # 将所有时间步的边索引堆叠为一个张量，形状为 (seq_len, num_edges, 2)
                edge_weight_tot = torch.stack(edge_weight_tot, dim=0)  # 将所有时间步的边权重堆叠为一个张量，形状为 (seq_len, num_edges)

                # 创建图数据
                data = Data(x=x_tot, edge_index=edge_index_tot, edge_attr=edge_weight_tot)
                # 计算跳数
                hop_targets.append(calculate_hops(graphs, seq_len=args.seq_len))
                # print("hop_targets", hop_targets[0].shape)

                # hop_targets = torch.stack(hop_targets, dim=0)  # 将跳数按时间步堆叠

                edge_label, edge_label_index = negative_sample(data)
                # print("edge label shape", edge_label.shape)


                # edge_label, edge_label_index = edge_label.to(device), edge_label_index.to(device)

                # 前向传播
                link_pred, hop_pred = model(data)
            hop_targets = torch.stack(hop_targets, dim=0)  # 将跳数按时间步堆叠
            loss_link = criterion_link(link_pred, edge_label)
            # hop_pred = hop_pred.view(seq_len, batch_size, num_nodes, num_nodes)  # (seq_len, batch_size, num_nodes, num_nodes)
            # print(f"hop_out shape after reshaping: {hop_pred.shape}")  # 打印检查 reshaped hop_out 的形状
            # print(f"hop_pred shape: {hop_pred.shape}")
            # print(f"hop_targets shape: {hop_targets.shape}")
            loss_hop = criterion_hop(hop_pred.squeeze(), hop_targets)

            # 总损失
            loss = loss_link + loss_hop
            losses_per_batch.append(loss)
            print("loss-----------------------------",loss)

            # 计算当前 batch 的平均 loss
            if len(losses_per_batch) > 0:
                batch_loss = sum(losses_per_batch) / len(losses_per_batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度剪裁
                optimizer.step()

                epoch_loss += batch_loss.item()

        if len(losses_per_batch) > 0:
            losses.append(epoch_loss / len(train_loader))
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}')

            # 调整学习率
            scheduler.step(epoch_loss / len(train_loader))

    # 绘制损失值变化图
    plot_loss(losses)

    # 在测试集上进行评估
    test(model, test_loader)

    # 链路预测评估
    evaluate_model(model)


# 测试函数定义

def test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, edge_indices, edge_weights in test_loader:
            batch_size = x.shape[0]
            for i in range(batch_size):
                data = Data(x=x[i].view(-1, x.size(-1)).to(device), edge_index=edge_indices[i].to(device),
                            edge_attr=edge_weights[i].to(device))
                edge_label, edge_label_index = negative_sample(data)
                edge_label, edge_label_index = edge_label.to(device), edge_label_index.to(device)
                pred = model(data.x, data.edge_index, edge_label_index).view(-1).sigmoid()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(edge_label.cpu().numpy())
    # 计算 AUC
    auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds))
    print(f'Test AUC: {auc}')


def evaluate_model(model):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            link_out, hop_out = model(data)

            edge_label, edge_label_index = negative_sample(data)
            y_true.append(edge_label.cpu().numpy())
            y_pred.append(link_out[edge_label_index[0], edge_label_index[1]].cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # ROC-AUC和平均精度
    roc_auc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")


# 使用 Dijkstra 算法进行所有节点到所有节点的路径预测
def dijkstra_all_pairs(graph):
    num_nodes = graph.num_nodes
    all_paths = {}
    for target_node in range(num_nodes):
        for start_node in range(num_nodes):
            visited = [False] * num_nodes
            min_heap = [(0, start_node)]  # (距离, 节点)
            distances = {i: float('inf') for i in range(num_nodes)}
            distances[start_node] = 0
            path = {i: [] for i in range(num_nodes)}
            path[start_node] = [start_node]

            while min_heap:
                current_distance, current_node = heapq.heappop(min_heap)
                if visited[current_node]:
                    continue
                visited[current_node] = True

                if current_node == target_node:
                    all_paths[(start_node, target_node)] = path[current_node]
                    break

                neighbors = graph.edge_index[1][graph.edge_index[0] == current_node]
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    distance = current_distance + 1  # 假设每条边的距离为 1
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(min_heap, (distance, neighbor))
                        path[neighbor] = path[current_node] + [neighbor]

            # 如果没有到达目标节点的路径
            if (start_node, target_node) not in all_paths:
                all_paths[(start_node, target_node)] = []

    return all_paths


# 比较 GNN 和 Dijkstra 的最短路径预测
def predict_all_pairs_routing_with_cache(model, data):
    model.eval()
    path_cache = {}
    with torch.no_grad():
        link_distances, hop_predictions = model(data)
        distances = link_distances.squeeze()
        hops = hop_predictions.squeeze()
        for target_node in range(data.num_nodes):
            for node in range(data.num_nodes):
                if (node, target_node) in path_cache:
                    continue
                current_node = node
                path = [current_node]
                total_hops = 0
                while current_node != target_node:
                    # 如果当前节点已在路径缓存中，扩展路径并返回
                    if (current_node, target_node) in path_cache:
                        path.extend(path_cache[(current_node, target_node)][1:])  # 避免重复当前节点
                        total_hops += len(path_cache[(current_node, target_node)][1:])
                        break

                    # 选择到目标节点通信距离最短的下一跳节点
                    neighbors = data.edge_index[1][data.edge_index[0] == current_node]
                    if len(neighbors) == 0:
                        print(f"从节点 {current_node} 到达节点 {target_node} 没有可用路径")
                        break
                    next_hop = neighbors[torch.argmin(distances[neighbors])].item()
                    path.append(next_hop)
                    total_hops += 1
                    current_node = next_hop

                # 缓存路径中所有节点的计算路径和跳数
                for i in range(len(path)):
                    if (path[i], target_node) not in path_cache:
                        path_cache[(path[i], target_node)] = path[i:]

    print(f'所有路径： {path_cache}')
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
    for key in dijkstra_paths:
        dijkstra_path = dijkstra_paths[key]
        gnn_path = gnn_paths.get(key, [])
        if dijkstra_path == gnn_path:
            correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f'GNN 路径预测的准确度: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    train_model()
    train_loader, test_loader, num_nodes = process_data_from_file('data/trimmed_data_all.csv', 8, 32)
    for x, edge_indices, edge_weights in test_loader:
        data = Data(x=x[0].view(-1, x.size(-1)).to(device), edge_index=edge_indices[0].to(device), edge_attr=edge_weights[0].to(device))
        compare_routing_accuracy(GNNRoutingModel(2, 256, 1).to(device), data)
