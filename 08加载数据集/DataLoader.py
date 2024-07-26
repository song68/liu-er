import torch
import numpy as np
from torch.utils.data import Dataset # Dataset是一个抽象类，不能实例化，只能被其他子类继承
from torch.utils.data import DataLoader # 加载数据
 

# 1.准备数据集
class DiabetesDataset(Dataset):                         # DiabetesDataset继承自抽象类DataSet
    def __init__(self, filepath):                       # filepath文件来自什么地方
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]                          # shape(多少行，多少列)，程序是取行
        self.x_data = torch.from_numpy(xy[:, :-1])      
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):                       # DiabetesDataset实例化之后，对象能够支持下标操作
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):                                  # 使用len()方法后，能够把数据集的数据条数返回
        return self.len
 

# 2.实例化dataset对象
dataset = DiabetesDataset('./diabetes.csv')


# 3.使用DataLoader加载数据
train_loader = DataLoader(dataset=dataset,  # dataSet对象 
                            batch_size=32,  # 每个batch的条数
                            shuffle=True,   # 是否打乱
                            num_workers=4)  # 在读数据构成batch时，是否要并,多线程一般设置为4和8
 

# 4.设计模型类
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
 
model = Model()
 

# 5.构造损失函数和优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 

# 6.循环计算前馈 反向传播 更新权值
if __name__ == '__main__':                          # windows下问题，不写报错（封装到函数里面或者if语句里解决）
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # enumerate获得当前迭代的次数，0表示从下标为0开始，train_loader中xy放入data里面，train_loader 是先shuffle后mini_batch
            # 准备数据
            inputs, labels = data                   # 把输入x和标签y拿出，（dataset每次拿出来都是x[i] y[i]，dataloader每次拿到一组之后，就是dataset每次拿到一个数据样本，dataloader每次根据batch的数量把x和y变成2个矩阵，dataloader会自动的把他们转成Tensor）取出一个batch 
            # 前馈
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            # 反馈
            optimizer.zero_grad()
            loss.backward()
            # 更新
            optimizer.step()