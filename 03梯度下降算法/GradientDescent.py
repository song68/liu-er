# 梯度下降
import matplotlib.pyplot as plt
 
# 训练数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# 初始权重
w = 1.0
 
# 前馈
def forward(x):
    return x*w

# 定义损失代价函数MSE
def cost(xs, ys): # 要进行求和，所以把x,y所有数据都拿进来
    cost = 0
    # 对损失值进行求和
    for x, y in zip(xs,ys):
        # 计算预测
        y_pred = forward(x)
        # 计算损失值
        cost += (y_pred - y)**2
    # 除去样本的数量，求均值
    return cost / len(xs)

# 定义梯度，更新w朝着梯度的最优方向搜索
def gradient(xs,ys):
    grad = 0
    for x, y in zip(xs,ys):
        grad += 2*x*(x*w - y)
    return grad / len(xs)
 
# 记录
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))

# 训练
for epoch in range(1000):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    # 更新权值
    w-= 0.001 * grad_val  # 0.01 learning rate
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

# 绘图
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show() 