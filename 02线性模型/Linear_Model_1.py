import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置numpy打印选项，确保打印完整数组输出
np.set_printoptions(threshold=np.inf)

#这里设函数为y=3x+2
x_data = [1.0,2.0,3.0]
y_data = [5.0,8.0,11.0]

# 前馈
def forward(x):
    return x * w + b

# 损失
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2

# 记录训练参数
mse_list = []

# 可设置步长为1进行分析
W=np.arange(0.0,4.1,0.1) # 生成从0.0到4.0（含4.0），步长为0.1的浮点数序列
B=np.arange(0.0,4.1,0.1)

# print(W)
# print(B)

# 生成网格采样点
[w,b]=np.meshgrid(W,B)
print([w,b])

# 训练
l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)     # 自动计算了整个列表，不用循环
    print(y_pred_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

# 绘图
fig = plt.figure()   # 创建一个新的图形对象
ax = fig.add_subplot(111, projection='3d')    # 添加一个3D子图
ax.plot_surface(W, B, l_sum/3, cmap='viridis')    # 绘制3D表面图

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

plt.show()