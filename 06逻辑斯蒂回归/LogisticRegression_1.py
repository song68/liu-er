import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
 
# 1.准备数据集----变为0,0,1为第0类和第一类
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
 
# 2.设计模型类-----加logistic的非线性变换
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        # y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()
 
# 3.构造损失函数和优化器----BCELoss
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加
# 二分类的交叉熵
criterion = torch.nn.BCELoss(size_average = False)          
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 
# 4.循环计算前馈 反向传播 更新权值
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)


x = np.linspace(0, 10, 200) # 使用 np.linspace 函数生成一个从 0 到 10 的 200 个均匀间隔的点。
print(x)
x_t = torch.Tensor(x).view((200, 1)) # 将 x 转换为 PyTorch 的张量，并重塑为形状为 (200, 1) 的二维张量。
print(x_t)
y_t = model(x_t)
print(y_t)
y = y_t.data.numpy() # 将 PyTorch 张量 y_t 转换为 NumPy 数组 y。y_t.data 取出张量的数据部分，.numpy() 将其转换为 NumPy 数组。
print(y)
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5] ,c='r') # 绘制一条从 (0, 0.5) 到 (10, 0.5) 的红色水平线，表示某个基准概率值（例如，通过的概率为 0.5）。
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid() # 显示网格线，方便观察图形。
plt.show() # 显示绘制的图形。