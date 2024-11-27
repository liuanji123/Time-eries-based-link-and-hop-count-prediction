# Usage

## create python virtual env

```bash
pip install --upgrade pip
python -m venv myenv
myenv\Scripts\activate
```

## cpu part

```bash
pip install -r req.txt
```

## 实验结果分析

现在单纯进行跳数预测来实现在准确率在一定范围的情况下使速度更快，但是现在hop_pred文件中GNN 所有节点到所有节点路径预测时间: Dijkstra 所有节点到所有节点路径预测时间: GNN 所有节点到所有节点路径预测时间: 0.7884466648101807 秒，其中神经网络加载所需时间: 0.38558435440063477 秒，bb计算所需时间: 0.4028623104095459 秒，Dijkstra 所有节点到所有节点路径预测时间: 0.03347444534301758 秒，神经网络加载的时间已经基本超过dijkstra的时间，而且准确率也只有30%，感觉已经要寄了，拟合的效果也不是很好

为什么
GNN 路径预测的准确度: 11.52%
GNN 距离预测的准确度: 5.49%
不应该路径预测对了，距离就是对的吗
