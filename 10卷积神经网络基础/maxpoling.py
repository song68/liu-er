import torch
input = [3,4,6,5,
        2,4,6,8,
        1,6,7,8,
        9,7,4,6,
]
input = torch.Tensor(input).view(1, 1, 4, 4)

# 最大池化----当kernel_size=2时，默认 stride = 2
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output)
