import torch#Pytorch主库，包含神经网络、张量计算、自动求导功能
from torch.utils.data import DataLoader#数据加载器，将数据集分批送给训练模型，减少构造循环函数搬运数据的复杂性
from torchvision import transforms#图像预处理工具，将图片转化为张量
from torchvision.datasets import MNIST#torchvision封装好的MNIST数据集
import matplotlib.pyplot as plt#画图工具，最后显示手写数字图片和预测结果
import os#设置环境变量操作
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#保证如果OpenMP(Open Muti-Processing，多线程并行操作，让程序高效利用多核CPU的并行工具)运行库重复时，也允许执行(影响当前Python进程)
#"KMP_DUPLICATE_LIB_OK": "KMP"：Intel OpenMP相关前缀; "DUPLICATE":重复; "LIB":库; "OK":允许
import random
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

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

from torch.utils.data import DataLoader , random_split
from torchvision.datasets import MNIST
from torchvision import transforms

def get_data_loaders(batch_size=64):#拿到数据集并包装成Dateloader
    to_tensor = transforms.Compose([transforms.ToTensor()])#数据预处理，将图片从PIL图像或者numpy数组转换成Pytorch张量，同时将像素值从0~255缩放成0~1
    
    full_train_set=MNIST("data",train=True,transform=to_tensor,download=True)
    test_set = MNIST("data",train=False,transform=to_tensor,download=True)

    train_size=int(0.9*len(full_train_set))
    val_size=len(full_train_set)-train_size

    train_set,val_set=random_split(full_train_set,[train_size,val_size])

    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

    return train_loader,val_loader,test_loader

def evaluate(loader, net):
    net.eval()
    total_loss=0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x=x.view(x.size(0),-1)
            output = net(x)
            loss=torch.nn.functional.nll_loss(output,y)

            total_loss +=loss.item()*x.size(0)
            pred = output.argmax(dim=1)
            correct+=(pred == y).sum().item()
            total +=x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss,acc

def train_one_epoch(loader,net,optimizer):
    net.train()
    total_loss = 0.0
    total_samples = 0

    for x,y in loader:
        x=x.view(x.size(0),-1)

        optimizer.zero_grad()
        output = net(x)
        loss = torch.nn.functional.nll_loss(output,y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*x.size(0)
        total_samples += x.size(0)

    return total_loss/total_samples
def main():
    set_seed(42)

    train_data,val_data,test_data = get_data_loaders(batch_size=64)
    net = Net()
    
    initial_val_loss,initial_val_acc = evaluate(val_data,net)
    print(f"initial validation loss: {initial_val_loss:.4f}")
    print(f"initial validation accuracy: {initial_val_acc:.4f}")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(5):
        train_loss = train_one_epoch(train_data,net,optimizer)
        val_loss,val_acc = evaluate(val_data,net)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc>best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(),"best_mnist_mlp.pth")
            print("Best model saved.")

        print(f"Epoch {epoch+1}:train+loss={train_loss:.4f},val_loss={val_loss:.4f},val_acc={val_acc:.4f}")

    net.load_state_dict(torch.load("best_mnist_mlp.pth",weights_only=True))
    net.eval()

    test_loss,test_acc = evaluate(test_data,net)
    print(f"final test loss: {test_loss:.4f}")
    print(f"final test accuracy:{test_acc:.4f}")

    net.eval()
    with torch.no_grad():
        for (n, (x, _)) in enumerate(test_data):
            if n > 3:
                break
            predict = torch.argmax(net(x[0].view(-1, 28*28)))

            plt.figure(n)
            plt.imshow(x[0].view(28, 28))
            plt.title("prediction: " + str(int(predict.item())))
    plt.show()
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(train_losses,label="train loss")
    plt.plot(val_losses,label="val loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(val_accs,label="val accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.show()


if __name__ == "__main__":
    main()