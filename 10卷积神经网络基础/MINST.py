import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)        # 卷积层
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)       # 卷积层
        self.pooling = torch.nn.MaxPool2d(2)                      # 池化层
        self.fc = torch.nn.Linear(320, 10)                        # 全连接层
 
 
    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)     # batchsize样本数量
        x = F.relu(self.pooling(self.conv1(x)))   # 先卷积，再池化，最后激活
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) # 变成全连接网络需要的输入   -1 此处自动算出的是320
        x = self.fc(x)
        return x
  
model = Net()
 

# 3.构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 

# 4.训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
 

# 5.预测 
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()