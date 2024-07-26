import torch
# 用于数据集的加载
from torchvision import transforms              # 主要针对图像进行各种各样的处理
from torchvision import datasets
from torch.utils.data import DataLoader
# 使用激活函数relu
import torch.nn.functional as F
# 优化器
import torch.optim as optim


# 1.准备数据集
batch_size = 64
# 归一化,均值和方差
transform = transforms.Compose([   # Compose可以把中括号里一系列对象构成一个pipeline一样的处理（拿到一个原始图像，先用ToTensor把输入图像转变成一个pytorch里的张量，取值变0-1。第二步做Normalize）
    transforms.ToTensor(),  # Convert the PIL Image to Tensor.
    transforms.Normalize((0.1307,), (0.3081,)) # 归一化 （均值）（方差）
]) 

train_dataset = datasets.MNIST(root='../data/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../data/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# 2.设计模型类
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 5 层
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
 
    # 前馈
    def forward(self, x):
        x = x.view(-1, 784)     # -1其实就是自动获取mini_batch,使用wiew展开张量为向量
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)       # 最后一层不做激活，不进行非线性变换
 
 
model = Net()
 

# 3.构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()                             # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)    # SGD优化器


# 4.训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):   # 0 是可选参数，表示索引的起始值，默认情况下索引从0开始。
        inputs, target = data
        optimizer.zero_grad()    # 梯度清零位置，再循环开始和前馈后反馈前效果都一样，只要在进行反向传播计算梯度前，将上一轮梯度清零就可以

        # 前馈 
        outputs = model(inputs)     
        loss = criterion(outputs, target)
        # 反馈 
        loss.backward()
        # 更新
        optimizer.step()

        # 每300个数据打印一次损失率
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

            
# 5.预测
def test():           # 不需要进行反向传播
    correct = 0       # 正确多少
    total = 0         # 总数多少
    with torch.no_grad():   # 不计算梯度
        for data in test_loader:   # 从test_loader拿数据
            images, labels = data
            # 预测 结果为一个batch长度×Feature个数的张量
            outputs = model(images)
            # torch.max的返回值有两个，第一个是每一行的最大值是多少，第二个是每一行最大值的下标(索引)是多少。
            _, predicted = torch.max(outputs.data, dim=1) # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()