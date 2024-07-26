import torch
in_channels, out_channels= 5, 10            # 输入输出通道
width, height = 100, 100                    # 输入图像的宽高
kernel_size = 3                             # 卷积核的大小
batch_size = 1                              # batch的个数(pytorch所有输入数据必须是小批量的)

input = torch.randn(batch_size,             # 随机生成一个张量输入，模拟图像
                    in_channels,            # input = (1,5,100,100)
                    width,                  # 对输入没有要求，除了in_channels必须要为5 否则卷积层不起作用
                    height)

# 定义卷积（必须设置下面三个变量）
conv_layer = torch.nn.Conv2d(   in_channels,             # 输入通道数量
                                out_channels,            # 输出通道数量
                                kernel_size=kernel_size) # 卷积核大小    3*3 也可以给元组 ex：5*3

output = conv_layer(input)                  # 做卷积

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape) (输出通道*输入通道*卷积核大小)
