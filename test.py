import torch#Pytorch主库，包含神经网络、张量计算、自动求导功能
from torch.utils.data import DataLoader#数据加载器，将数据集分批送给训练模型，减少构造循环函数搬运数据的复杂性
from torchvision import transforms#图像预处理工具，将图片转化为张量
from torchvision.datasets import MNIST#torchvision封装好的MNIST数据集
import matplotlib.pyplot as plt#画图工具，最后显示手写数字图片和预测结果
import os#设置环境变量操作
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#保证如果OpenMP(Open Muti-Processing，多线程并行操作，让程序高效利用多核CPU的并行工具)运行库重复时，也允许执行(影响当前Python进程)
#"KMP_DUPLICATE_LIB_OK": "KMP"：Intel OpenMP相关前缀; "DUPLICATE":重复; "LIB":库; "OK":允许

class Net(torch.nn.Module):#定义Net的神经网络类，继承Pytorch的nn.Moudle，确保Pytorch知道模型层数和训练参数

    def __init__(self):#定义网络层(定义网络结构)
        super().__init__()#调用初始化方法
        self.fc1 = torch.nn.Linear(28*28, 64)#构建四层传播层
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)#基础多层感知机MLP
    
    def forward(self, x):#定义向前传播(数据流动)
        x = torch.nn.functional.relu(self.fc1(x))#利用relu整流函数引入非线性变换
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)#利用log_softmax变成概率分布并取对(与后面损失函数torch.nn.functional_loss对应)
        return x


def get_data_loader(is_train):#拿到数据集并包装成Dateloader
    to_tensor = transforms.Compose([transforms.ToTensor()])#数据预处理，将图片从PIL图像或者numpy数组转换成Pytorch张量，同时将像素值从0~255缩放成0~1
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
