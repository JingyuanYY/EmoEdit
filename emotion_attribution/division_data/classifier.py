import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):  # 假设有10个分类
        super(SimpleCNN, self).__init__()
        # 第一层卷积层：输入通道数为4，输出通道数为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        # 第二层卷积层：输入通道数为16，输出通道数为32，卷积核大小为3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 池化层：使用最大池化，池化核大小为2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层：输入维度为32 * 16 * 16，输出维度为128
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        # self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # 最后一层全连接层：将128个特征映射到num_classes个输出
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [100, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [100, 32, 16, 16]
        x = x.view(-1, 32 * 16 * 16)  # 展平：100 x (32 * 16 * 16)
        # x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))  # 全连接层1
        x = self.fc2(x)  # 输出层
        return x


if __name__ == '__main__':
    """Test the Classifier"""
    # model = Classifier(768, 8)
    # data = torch.randn(32, 768)
    # output1 = model(data).squeeze(1)
    # output2 = model(data.squeeze(1))
    # print(output1, output2)
    # print(output1.shape, output2.shape)
    # print(output1 == output2)
    """Test the simpleCNN"""
    model = SimpleCNN(8)
    data = torch.randn(100, 4, 64, 64)
    output = model(data)
    print(output)
    print(output.shape)
