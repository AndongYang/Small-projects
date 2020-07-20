import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import os
#import nni

#使用ResNet18尝试后发现mac运行过于缓慢
#综合考虑电脑性能与效果选择使用alexnet

#定义alexnet主体
class my_ResNet18(nn.Module):
    def __init__(self):
        #尝试使用python3方法，直接super().xxx
        #super(my_LeNet, self).__init__()
        super().__init__()
        self.in_f = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.cla = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 2),
        )


    def forward(self, x):
        x = self.in_f(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.cla(x)

        return x  

#处理无显卡情况，由于使用的电脑中gpu0为集显，因此将使用的显卡设为独显gpu1。
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICE']='1'


#nni获得参数
#params = nni.get_next_parameter()
GLOBAL_LR = 0.0001
GLOBAL_BATCH_SIZE = 32
GLOBAL_EPOCH = 10

#准备数据集
print("Loading dataset.")
#处理原始图片
my_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

dataset_train = torchvision.datasets.ImageFolder(root = './data/train/', transform=my_transform)
dataset_test = torchvision.datasets.ImageFolder(root = './data/test/', transform=my_transform)

train_data = torch.utils.data.DataLoader(dataset_train, batch_size=GLOBAL_BATCH_SIZE, shuffle=True)
test_data = torch.utils.data.DataLoader(dataset_test, batch_size=GLOBAL_BATCH_SIZE, shuffle=True)

#具体化网络
print("Creating net.")
my_net = my_ResNet18().to(GLOBAL_DEVICE)

#定义交叉熵损失函数与adam优化器
Loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(my_net.parameters(), lr = GLOBAL_LR)
#opt = torch.optim.SGD(my_net.parameters(), lr = GLOBAL_LR)

if os.path.exists('net.pth'):
    my_net.load_state_dict(torch.load('net.pth'))
    print('Loading net')
else:
    print('Not loading net')

if __name__ == '__main__':
    print("Starting main loop.")
    for i in range(GLOBAL_EPOCH):
        for loop, data in enumerate(train_data):
            #准备输入数据
            source_imgs, labels = data
            source_imgs = source_imgs.to(GLOBAL_DEVICE)
            labels = labels.to(GLOBAL_DEVICE)
            #梯度归零
            opt.zero_grad()
            #通过网络计算结果
            res = my_net(source_imgs)
            #通过Loss函数计算损失
            loss = Loss(res, labels)
            #对损失进行梯度下降
            loss.backward()
            #根据损失调整参数
            opt.step()
            
            #对训练情况进行输出
            if loop%100 == 0:
                loss_rem = loss.detach()
                print("Epoch:{:d}, Loops:{:d}, Loss:{:.4f}".format(i,loop,loss_rem))
                torch.save(my_net.state_dict(),'net.pth')


#对训练结果模型进行测试
print("Begainning test.")
my_net.eval()
currect_res = 0
total_res = 0
for loop, data in enumerate(test_data):
    source_imgs,labels = data
    source_imgs = source_imgs.to(GLOBAL_DEVICE)
    labels = labels.to(GLOBAL_DEVICE)

    test_res = my_net(source_imgs)
    #print("test_res.size():{} ".format(str(test_res.size())))

    #统计准确率，即结果中与给定标签相同比例
    #将每个batch中的网络输出取最大值做为预测结果,结果表示为最大值位置，因此只需要第二个返回数据
    _, predicted_res = torch.max(test_res, 1)
    currect_res += (predicted_res == labels).sum().item()
    total_res += labels.size(0)

print("Avg acc:{:.4f}".format(1.0*currect_res/total_res))

    #nni传递结果
    #nni.report_final_result(test_acc)



