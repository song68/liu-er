# 绘图
import numpy as np              
import matplotlib.pyplot as plt 
 
# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 定义模型 前馈
def forward(x):
    return x*w

# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# 保存权重和损失
w_list = []
mse_list = []

# 穷举法
for w in np.arange(0.0, 4.1, 0.1): # 生成从0.0到4.0（含4.0），步长为0.1的浮点数序列
    print("w=", w)
    l_sum = 0
    # 将对象中对应的元素打包成一个个元组,然后返回由这些元组组成的列表
    for x_val, y_val in zip(x_data, y_data):
        # 计算预测
        y_pred_val = forward(x_val)
        # 计算损失
        loss_val = loss(x_val, y_val)
        # 求和
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    # 添加到列表
    w_list.append(w)
    mse_list.append(l_sum/3)

# 绘制图形
plt.plot(w_list, mse_list) # 绘制w_list和mse_list之间的关系图
plt.ylabel('Loss') # 设置y轴标签
plt.xlabel('w') # 设置x轴标签
plt.show() # 显示图形


# t=zip(x_data, y_data)
# print(list(t))

'''
    x_data和y_data是两个等长的列表。
    zip(x_data, y_data)将这两个列表打包成一个迭代器,每次迭代时会返回一个元组,元组的第一个元素来自x_data,第二个元素来自y_data。
    for x_val, y_val in zip(x_data, y_data):这一行代码表示我们将依次从zip生成的迭代器中获取元组并,将元组的第一个值赋给x_val,第二个值赋给y_val。
    在循环体内,我们对x_val和y_val进行操作,比如计算它们的和并,打印结果。
'''