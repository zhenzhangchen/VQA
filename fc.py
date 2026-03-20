from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch
# 新增梯度反转层模块
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
class BiasBranchMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 512,  # 简化隐藏层
                 activation: str = 'leaky_relu',  # 更改为LeakyReLU
                 dropout: float = 0.3,  # 降低Dropout比例
                 use_weight_norm: bool = False):  # 可选是否使用权重归一化
        super(BiasBranchMLP, self).__init__()
        
        # 简化结构：输入 -> 隐藏层 -> 输出
        layers = []
        
        # 第1层：输入到隐藏层
        layers.append(
            weight_norm(nn.Linear(input_dim, hidden_dim), dim=None) 
            if use_weight_norm else 
            nn.Linear(input_dim, hidden_dim)
        )
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        # 第2层：隐藏层到输出
        layers.append(
            weight_norm(nn.Linear(hidden_dim, output_dim), dim=None) 
            if use_weight_norm else 
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.net = nn.Sequential(*layers)
        
        # 梯度反转层（整合到BiasBranch内部）
        self.grl = GradientReversal(alpha=1.0)  # 需提前定义GradientReversal类

    def forward(self, x):
        x = self.grl(x)  # 应用梯度反转
        return self.net(x)

# relu(linear(1024,1024))-->relu(linear(1024,1024))-->linear(1024,2274)
class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 dimensions,
                 activation='relu',
                 dropout=0.5):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([weight_norm(nn.Linear(input_dim, dimensions[0]),dim=None)])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(weight_norm(nn.Linear(din, dout),dim=None))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears) - 1):
                x = nn.functional.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = nn.functional.dropout(x, self.dropout, training=self.training)
        return x


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        # return nn.ReLU(weight_norm(x,dim=None))


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
