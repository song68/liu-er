import torch
import torch.nn.functional as F
 
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