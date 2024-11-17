
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
    # print(f"num_nodes: {num_nodes}")
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
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 链路预测的全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 结合高阶和低阶特征
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 链路预测输出
        # 跳数预测的全连接层
        self.fc_hop = nn.Linear(hidden_dim * 2, num_nodes)  # 调整输出维度

    def encode(self, x, edge_index):
        x = x.view(-1, x.size(-1))
        x = self.gat1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = z[edge_label_index[0]], z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        seq_len = data.x.size(0)  # 获取序列长度

        x = data.x  # (seq_len, num_nodes, feature_dim)
        edge_indices = data.edge_index  # (seq_len, num_edges, 2)
        edge_weights = data.edge_attr  # (seq_len, num_edges)

        # 初始化输出
        link_out = []
        hop_out = []

        # 对每个时间步的图分别进行卷积操作
        for t in range(seq_len):
            x_t = x[t]  # 将 x[t] 调整为二维张量 (num_nodes, feature_dim)
            edge_index_hop = edge_indices[t]
            edge_attr = edge_weights[t]
            # print(edge_index.shape, edge_attr.shape)

            # 图卷积操作
            x_low = F.relu(self.conv1(x_t, edge_index_hop, edge_attr))
            x_low = F.relu(self.conv2(x_low, edge_index_hop, edge_attr))

            x_high = F.relu(self.conv1(x_t, edge_index_hop, edge_attr))
            x_high = F.relu(self.conv2(x_high, edge_index_hop, edge_attr))

            # 特征交叉
            x_combined = torch.cat([x_low, x_high], dim=-1)  # 通过拼接交叉特征

            # 链路预测
            # link_pred = F.relu(self.fc1(x_combined))
            # link_out.append(self.fc2(link_pred))  # 每个时间步的链路预测输出

            # 跳数预测
            hop_out.append(self.fc_hop(x_combined))  # 每个时间步的跳数预测输出

        edge_index_permuted = data.edge_index.permute(1, 0, 2)  # 通过 permute 交换维度顺序为 [2, 8, 80]
        edge_index_flattened = edge_index_permuted.reshape(2, -1)  # 将 [8, 80] 展平为 [640]，保留第一个维度

        # 链路预测
        z = self.encode(x, edge_index_flattened)
        link_pred = self.decode(z, edge_label_index)

        # link_out = torch.stack(link_out, dim=0)  # (seq_len, num_nodes, num_nodes)
        hop_out = torch.stack(hop_out, dim=0)  # (seq_len, num_nodes, num_nodes)

        return link_pred, hop_out


# 定义 Dijkstra 函数以计算最短路径和跳数
def dijkstra_shortest_hops(graph, source_node):
    G = to_networkx(graph, to_undirected=True)
    lengths = nx.single_source_shortest_path_length(G, source=source_node)
    return lengths


# 计算跳数标签的函数
def calculate_hops(graphs, seq_len):
    hop_targets = []  # 用于存储所有时间步的跳数标签

    # 遍历每个时间步
    for t in range(seq_len):
        graph_t = graphs[t]  # 选择第 t 个时间步的图数据
        # 创建一个零矩阵来保存该时间步所有节点对的跳数
        hop_matrix = torch.zeros((graph_t.num_nodes, graph_t.num_nodes), dtype=torch.float)

        # 对每个时间步的每个节点，计算跳数
        for source_node in range(graph_t.num_nodes):
            # 获取从源节点到所有其他节点的最短路径
            lengths = dijkstra_shortest_hops(graph_t, source_node)

            # 对于每个目标节点，获取跳数（若没有路径，设为inf）
            for target_node in range(graph_t.num_nodes):
                hop_matrix[source_node, target_node] = lengths.get(target_node, float('inf'))

        hop_targets.append(hop_matrix)  # 将当前时间步的跳数矩阵添加到列表中

    # 将列表中的跳数矩阵堆叠成一个三维张量
    return torch.stack(hop_targets, dim=0)  # 返回形状为 (seq_len, num_nodes, num_nodes) 的三维张量


# 计算负量采样
def negative_sample(data, neg_ratio= 3):
    edge_index_permuted = data.edge_index.permute(1, 0, 2)  # 通过 permute 交换维度顺序为 [2, 8, 80]
    edge_index_flattened = edge_index_permuted.reshape(2, -1)  # 将 [8, 80] 展平为 [640]，保留第一个维度
    edge_index_flattened = edge_index_flattened.long()
    num_neg_samples = edge_index_flattened.size(1)
    # 正负采样按照一个比例 neg_ratio=3
    # num_pos_samples = edge_index_flattened.size(1)
    # num_neg_samples = num_pos_samples * neg_ratio

    neg_edge_index = negative_sampling(
        edge_index=edge_index_flattened, num_nodes=data.x.size(1),
        num_neg_samples=num_neg_samples, method='sparse'
    )
    edge_label_index = torch.cat([edge_index_flattened, neg_edge_index], dim=-1).long()
    edge_label = torch.cat([torch.ones(num_neg_samples), torch.zeros(neg_edge_index.size(1))], dim=0)
    perm = torch.randperm(edge_label.size(0))
    edge_label = edge_label[perm]

    edge_label_index = edge_label_index[:, perm]
    return edge_label, edge_label_index


    # num_pos_samples = data.edge_index.size(1)
    # print(f'num_pos_samples : {num_pos_samples} ')
    # num_neg_samples = num_pos_samples * neg_ratio
    #
    # # edge_index 是三维的(seq_len, num_edges, 2), 遍历每个时间步
    # edge_label_index_list = []
    # for t in range(data.edge_index.size(0)):  # 如果数据包含多个时间步
    #     edge_index = data.edge_index[t]  # 选择第 t 个时间步的边
    #
    #     neg_edge_index = negative_sampling(
    #         edge_index, num_nodes=data.num_nodes,
    #         num_neg_samples=num_neg_samples, method='sparse'
    #     )
    #     edge_label_index_list.append(torch.cat([edge_index, neg_edge_index], dim=-1).long())
    #
    # edge_label_index = torch.cat(edge_label_index_list, dim=-1)
    # edge_label = torch.cat([torch.ones(num_pos_samples), torch.zeros(num_neg_samples)], dim=0)
    # perm = torch.randperm(edge_label.size(0))
    # edge_label = edge_label[perm]
    # edge_label_index = edge_label_index[:, perm]
    # return edge_label, edge_label_index

# 保存模型的训练结果
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# 加载保存的模型
def load_model(model, file_path):
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        model.eval()  # 切换到评估模式
        print(f"Model loaded from {file_path}")
    else:
        print(f"No saved model found at {file_path}. Proceeding with an untrained model.")


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
    model = GNNRoutingModel(input_dim=args.input_size,
                            hidden_dim=args.hidden_size,
                            output_dim=args.output_size,
                            num_nodes=num_nodes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion_link = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device)) # 这个很有用吗？

    criterion_hop = nn.MSELoss()  # 跳数预测的损失函数
    for epoch in range(30):
        print(f"Epoch {epoch + 1}, num_nodes: {num_nodes}")
        model.train()
        epoch_loss = 0
        epoch_link_loss = 0
        epoch_hop_loss = 0
        for x, edge_indices, edge_weights in train_loader:
            optimizer.zero_grad()

            batch_size = x.shape[0]
            seq_len = x.shape[1]  # 序列长度
            num_nodes = x.shape[2]  # 节点数量
            losses_link_per_batch = []
            losses_hop_per_batch = []
            losses_per_batch = []
            for i in range(batch_size):
                hop_targets = []  # 每个批次开始时初始化 hop_targets

                graphs = []  # 用于保存每个时间步的图数据
                edge_index_tot = []
                edge_weight_tot = []
                x_tot = []
                for t in range(seq_len):  # seq_len
                    edge_index = edge_indices[t][i]  # 获取第 t 个时间步的边
                    data = Data(x=x[i, t].view(-1, x.size(-1)).to(device),  # 获取第t个时间步的节点特征
                                edge_index=edge_indices[t][i].to(device),  # 获取第t个时间步的边
                                edge_attr=edge_weights[t][i].to(device))  # 获取第t个时间步的边权重
                    edge_index_tot.append(edge_indices[t][i].to(device))
                    edge_weight_tot.append(edge_weights[t][i].to(device))
                    x_tot.append(x[i, t].view(-1, x.size(-1)).to(device))
                    graphs.append(data)  # 将每个时间步的图数据保存到 graphs 中

                # 将列表转换为张量
                x_tot = torch.stack(x_tot, dim=0)  # 将所有时间步的节点特征堆叠为一个张量，形状为 (seq_len, num_nodes, feature_dim)
                edge_index_tot = torch.stack(edge_index_tot, dim=0)  # 将所有时间步的边索引堆叠为一个张量，形状为 (seq_len, num_edges, 2)
                edge_weight_tot = torch.stack(edge_weight_tot, dim=0)  # 将所有时间步的边权重堆叠为一个张量，形状为 (seq_len, num_edges)

                # 创建图数据
                data = Data(x=x_tot, edge_index=edge_index_tot, edge_attr=edge_weight_tot)

                # 跳数标签
                hop_targets.append(calculate_hops(graphs, seq_len=args.seq_len).to(device))
                # 链路标签
                edge_label, edge_label_index = negative_sample(data)
                edge_label, edge_label_index = edge_label.to(device), edge_label_index.to(device)

                # 前向传播
                link_pred, hop_pred = model(data, edge_label_index)

                loss_link = criterion_link(link_pred, edge_label)
                losses_link_per_batch.append(loss_link)

            # 单独计算链路预测的损失
            batch_link_loss = sum(losses_link_per_batch) / len(losses_link_per_batch)
            epoch_link_loss += batch_link_loss.item()

            # 单独计算跳数预测的损失
            hop_targets = torch.stack(hop_targets, dim=0)  # 将跳数按时间步堆叠
            loss_hop = criterion_hop(hop_pred.squeeze(), hop_targets)
            losses_hop_per_batch.append(loss_hop)
            batch_hop_loss = sum(losses_hop_per_batch) / len(losses_hop_per_batch)
            epoch_hop_loss += batch_hop_loss.item()

            # 总损失
            loss = loss_link * 4 + loss_hop
            losses_per_batch.append(loss)
            print("loss-----------------------------", loss)
            # 计算当前 batch 的平均 loss
            batch_loss = sum(losses_per_batch) / len(losses_per_batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度剪裁
            optimizer.step()
            epoch_loss += batch_loss.item()


        losses.append(epoch_link_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, link_Loss: {epoch_link_loss / len(train_loader)}')
        losses.append(epoch_hop_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, hop_Loss: {epoch_hop_loss / len(train_loader)}')
        losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, all_Loss: {epoch_loss / len(train_loader)}')

        # 调整学习率
        scheduler.step(epoch_loss / len(train_loader))

    # 保存训练后的模型
    # save_model(model, "./data/trained_model.pth")
    #     # 绘制损失值变化图
    plot_loss(losses)

    # 在测试集上进行评估
    test(model, test_loader)


# 测试函数定义
def test(model, test_loader):
    # 加载模型
    # load_model(model, "./data/trained_model.pth")
    model.eval()
    all_link_preds = []
    all_link_labels = []
    all_hop_preds = []
    all_hop_labels = []
    with torch.no_grad():
        for x, edge_indices, edge_weights in test_loader:

            batch_size = x.shape[0]
            seq_len = x.shape[1]  # 序列长度
            num_nodes = x.shape[2]  # 节点数量

            losses_per_batch = []
            for i in range(batch_size):

                hop_targets = []  # 每个批次开始时初始化 hop_targets
                graphs = []  # 用于保存每个时间步的图数据
                edge_index_tot = []
                edge_weight_tot = []
                x_tot = []
                for t in range(seq_len):  # seq_len
                    edge_index = edge_indices[t][i]  # 获取第 t 个时间步的边
                    data = Data(x=x[i, t].view(-1, x.size(-1)).to(device),  # 获取第t个时间步的节点特征
                                edge_index=edge_indices[t][i].to(device),  # 获取第t个时间步的边
                                edge_attr=edge_weights[t][i].to(device))  # 获取第t个时间步的边权重
                    edge_index_tot.append(edge_indices[t][i].to(device))
                    edge_weight_tot.append(edge_weights[t][i].to(device))
                    x_tot.append(x[i, t].view(-1, x.size(-1)).to(device))
                    graphs.append(data)  # 将每个时间步的图数据保存到 graphs 中

                # 将列表转换为张量
                x_tot = torch.stack(x_tot, dim=0)  # 将所有时间步的节点特征堆叠为一个张量，形状为 (seq_len, num_nodes, feature_dim)
                edge_index_tot = torch.stack(edge_index_tot, dim=0)  # 将所有时间步的边索引堆叠为一个张量，形状为 (seq_len, num_edges, 2)
                edge_weight_tot = torch.stack(edge_weight_tot, dim=0)  # 将所有时间步的边权重堆叠为一个张量，形状为 (seq_len, num_edges)

                # 创建图数据
                data = Data(x=x_tot, edge_index=edge_index_tot, edge_attr=edge_weight_tot)

                hop_targets.append(calculate_hops(graphs, seq_len).to(device))

                edge_label, edge_label_index = negative_sample(data)
                edge_label, edge_label_index = edge_label.to(device), edge_label_index.to(device)
                # print(f"edge_label: {edge_label}")

                link_pred, hop_pred = model(data, edge_label_index)
                all_link_preds.append(link_pred.cpu().numpy())
                all_link_labels.append(edge_label.cpu().numpy())

            hop_targets = torch.stack(hop_targets, dim=0)  # 将跳数按时间步堆叠
            all_hop_preds.append(hop_pred.cpu().numpy())
            all_hop_labels.append(hop_targets.cpu().numpy())

    print(f"Shape of all_link_preds: {np.array(all_link_preds).shape}")
    print(f"Shape of all_link_labels.shape: {np.array(all_link_labels).shape}")
    print(f"Shape of all_link_labels: {np.array(all_link_labels)}")

    print(f"Shape of all_hop_preds: {np.array(all_hop_preds).shape}")
    print(f"Shape of all_hop_labels: {np.array(all_hop_labels).shape}")

    # 对于每个时间步，计算 AUC
    hop_preds_flat = np.concatenate([pred.flatten() for pred in all_hop_preds], axis=0)  # 展平所有预测
    hop_labels_flat = np.concatenate([label.flatten() for label in all_hop_labels], axis=0)  # 展平所有标签

    print(f"hop_preds_flat shape: {hop_preds_flat.shape}")
    print(f"hop_labels_flat shape: {hop_labels_flat.shape}")

    auc_link = roc_auc_score(np.concatenate(all_link_labels), np.concatenate(all_link_preds))
    # auc_link = roc_auc_score(link_labels_flat, link_preds_flat)
    mse_hop = mean_squared_error(hop_labels_flat, hop_preds_flat)

    print(f'Test link AUC: {auc_link}')
    print(f'Test hop mean: {mse_hop}')


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
        compare_routing_accuracy(GNNRoutingModel(2, 256, 1, num_nodes).to(device), data)
