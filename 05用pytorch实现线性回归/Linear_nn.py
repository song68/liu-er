# 线性回归
import torch

# 1.准备数据集
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]]) 
# print(x_data)


# 2.设计模型类
class LinearModel(torch.nn.Module):
    # 构造
    def __init__(self):
        super(LinearModel, self).__init__()     
        # super父类，调用父类的构造，第一个参数是自己定义类的名称，第二个self，调用父类的__init__()方法
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        # torch.nn.Linear是pytorch的一个类，类后加括号，实际上在构造一个对象
        # Linear对象包含了权重和偏置两个tensor，将来可以直接用Linear来完成权重*输入+偏置的计算
        # Linear继承自Module，所以能够自动进行反向传播
        self.linear = torch.nn.Linear(1, 1)
    
    # 前馈
    def forward(self, x):
        y_pred = self.linear(x)  # 上个函数定义的，加括号表示实现了一个可调用的对象
        return y_pred

# 实例化----可以被直接调用
model = LinearModel()


# 3.构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average = False)
# criterion = torch.nn.MSELoss(reduction = 'sum')             # 使用MSE也可以自己设计
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)  # 优化器 / model.parameters()自动完成参数的初始化操作


# 4.循环计算前馈 反向传播 更新权值
for epoch in range(1000):
    y_pred = model(x_data)              # forward:predict
    loss = criterion(y_pred, y_data)    # forward: loss
    print(epoch, loss.item())
 
    optimizer.zero_grad()               # 梯度会积累 所以及时清0
    loss.backward()                     # backward: autograd 自动计算梯度
    optimizer.step()                    # update 参数 即更新w和b的值
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)