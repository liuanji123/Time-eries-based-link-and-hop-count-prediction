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

## gpu part

现在单纯进行跳数预测来实现在准确率在一定范围的情况下使速度更快，但是现在hop_pred文件中GNN 所有节点到所有节点路径预测时间: 53.42944264411926 秒，Dijkstra 所有节点到所有节点路径预测时间: 0.03649163246154785 秒差距过大，
