import numpy as np
import torch
import matplotlib.pyplot as plt
 
print
# 1.准备数据集
xy = np.loadtxt('./diabetes.csv', delimiter=',', dtype=np.float32) # loadtxt读取文件，后面加.gz也可读取。delimiter是分隔符，因为数据里面都是以逗号分隔，dtype指定数据类型
x_data = torch.from_numpy(xy[:, :-1])   # 取出前-1列 Feature（所有行）
y_data = torch.from_numpy(xy[:, [-1]])  # 取最后一列 label，用[]表示是一个矩阵，直接-1就是向量了。
 

# 2.设计模型类
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)    # 输入数据x的特征（列）是8维，x有8个特征，输出维度6维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()       # 激活函数，将其看作是网络的一层，而不是简单的函数使用
 
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))       # y hat
        return x
 

model = Model()
 

# 3.构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
 
epoch_list = []
loss_list = []


# 4.循环计算前馈 反向传播 更新权值
for epoch in range(100):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    epoch_list.append(epoch)
    loss_list.append(loss.item())
 
    # 反馈
    optimizer.zero_grad()
    loss.backward()
 
    # 更新
    optimizer.step()
 
 
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()