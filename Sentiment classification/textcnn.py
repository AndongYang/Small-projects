import torch
import time
import math
import datetime
import argparse
import logging
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import *

#初始化参数
parser = argparse.ArgumentParser(description='rnn')

parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_num', default=3, type=int, metavar='N',
                    help='epoch number')
parser.add_argument('--batch_size', default=50, type=int, metavar='BS',
                    help='batch size of training')
parser.add_argument('--kernel_size', default=[4], type=int, metavar='BS',
                    help='batch size of training')
parser.add_argument('--kernel_num', default=256, type=int, metavar='BS',
                    help='number of kernel')
parser.add_argument('-dropout', default=0.5, type=float, metavar='BS',
                    help='the probability for dropout [default: 0.5]')
parser.add_argument('--train_flag', default='train', type=str, 
                    help='train/test/validation')

#我的笔记本上gpu0是核显，1是独显
parser.add_argument('--gpu', default='1', type=str, help='GPU id to use.')
parser.add_argument('--max_gen_len', default=75, type=int, metavar='mgl',
                    help='lenth of res')
parser.add_argument('--embedding_dim', default=50, type=int, metavar='ed',
                    help='embedding_dim of text_cnn')
parser.add_argument('--models_save', default='./checkpoint.pth', type=str, 
                    metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

#获得参数
global args
args = parser.parse_args()
#初始化日志
logging.basicConfig(filename="./text_cnn.log", level=logging.ERROR)
#指定使用的显卡
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#GLOBAL_DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICE']=args.gpu
#记录开始时间
start_time = time.time()


class textcnn(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #输入词向量
        self.embed = nn.Embedding(vocab_size, args.embedding_dim)
        #卷积层
        self.convs = nn.ModuleList([nn.Conv2d(1,args.kernel_num,(K,args.embedding_dim)) for K in args.kernel_size]) ## 卷积层
        #池化层
        self.dropout = nn.Dropout(args.dropout)
        #全联接层
        self.fc = nn.Linear(len(args.kernel_size)*args.kernel_num, 2)
        
    def forward(self,x):
        x = self.embed(x)
        
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]
        
        x = torch.cat(x,1)
        
        x = self.dropout(x)
        output = self.fc(x)
        return output

class Avg_Loss():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.num = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val):
        self.num+=1
        self.sum+=val
        self.avg=self.sum/self.num
        return self.avg

avg_loss_log = Avg_Loss()

def train(dataloader, model, loss_cal, optimizer, epoch):
    '''Returen avg_loss'''
    model.train()

    for i, data in enumerate(dataloader):
        inputs, labels = data[0].long().to(GLOBAL_DEVICE), data[1].to(GLOBAL_DEVICE)
        #print(inputs)
        #如果GPU可用，将数据移入指定硬件
        #data = torch.from_numpy(np.array(data[0], dtype=np.float32))
        #if torch.cuda.is_available():
            # inputs = torch.as_tensor(np.array(inputs)).to(GLOBAL_DEVICE)
            # labels = torch.as_tensor(np.array(labels)).to(GLOBAL_DEVICE) 
        
        #labels = labels.view(-1)
        
        optimizer.zero_grad()
        cal_label = model(inputs)
        loss = loss_cal(cal_label, labels)
        loss.backward()
        optimizer.step()

        #每隔一定轮数输出一次结果
        avg_loss = avg_loss_log.update(loss.item())
        if i % args.print_freq == 0 or i == len(dataloader)-1:
            output_log("=> " 
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4f}\t'
                    .format(epoch, i, len(dataloader), batch_time=time.time() - start_time, 
                        loss=avg_loss), logging)

    return avg_loss

def eval_net(dataloader, model):
    model.eval()

    tp=0
    tn=0
    fn=0
    fp=0
    currect_res = 0
    total_res = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data[0].long().to(GLOBAL_DEVICE), data[1].to(GLOBAL_DEVICE)
        #labels = labels.view(-1)
        
        cal_label = model(inputs)
        predicted_res = torch.max(cal_label,1)[1]
        currect_res += (predicted_res == labels).sum().item()
        total_res += labels.size(0)

        # TP    predict 1 label 1
        tp += ((predicted_res == 1) & (labels == 1)).sum().item()
        # TN    predict 0 label 0
        tn += ((predicted_res == 0) & (labels == 0)).sum().item()
        # FN    predict 0 label 1
        fn += ((predicted_res == 0) & (labels == 1)).sum().item()
        # FP    predict 1 label 0
        fp += ((predicted_res == 1) & (labels == 0)).sum().item()

        #每隔一定轮数输出一次结果
        if i % args.print_freq == 0 or i == len(dataloader)-1:
            output_log("=> " 
                    'iter: [{0}/{1}]\t'
                    'Time {batch_time:.3f}\t'
                    'acc {res:.4f}\t'
                    .format(i, len(dataloader), batch_time=time.time() - start_time, 
                        res=currect_res/total_res), logging)

    percision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    F1 = 2.0 * recall * percision / (recall + percision)
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)

    return accuracy, percision, recall, F1


def output_log(data, logger=None):
    print("{}:{}".format(datetime.datetime.now(), data))
    if logger is not None:
        logger.critical("{}:{}".format(datetime.datetime.now(), data))

def main():
    #获取训练数据
    word2id = build_word2id('./Dataset/word2id.txt')
    word_vecs = build_word2vec("./Dataset/wiki_word2vec_50.bin", word2id, "word2vecs.txt")

    if args.train_flag == 'train':
        data_file_name = './Dataset/train.txt'
    elif args.train_flag == 'test':
        data_file_name = './Dataset/test.txt'
    elif args.train_flag == 'validation':
        data_file_name = './Dataset/validation.txt'
    else:
        output_log("=>unknown train flag.", logging)

    dataloader = DataLoader(text_cnn_Data(word_vecs, word2id, data_file_name),
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=2)

    output_log("=>loaded data.", logging)

    # tmp = 0
    # for i ,data in enumerate(train_dataloader):
    #     print("{}:{}".format(i, data))
    #     tmp+=1
    #     if tmp ==10:
    #         break
    # return

    #初始化网络模型
    print("len(word2id):{}".format(len(word2id)))
    model = textcnn(len(word2id)).to(GLOBAL_DEVICE)
    output_log("=>created net.", logging)

    #初始化优化器与损失函数
    loss_cal = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    #恢复checkpoint
    if os.path.isfile(args.models_save):
        output_log("=> loading checkpoint '{}'".format(args.models_save),
                    logging)
        checkpoint = torch.load(args.models_save)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        output_log("=> loaded checkpoint '{}' "
                    .format(args.models_save), logging)
    else:
        output_log("=> no checkpoint found at '{}'".format(args.models_save),logging)

    if args.train_flag == 'train':
        output_log("=>train or eval:'{}'".format('train'), logging)
        #记录最准确的模型，方便最后输出参数
        best_prec = math.inf

        #开始主循环
        for epoch in range(args.epoch_num):
            #训练网络
            prec = train(dataloader, model, loss_cal, optimizer, epoch)
            #更新lr
            lr_scheduler.step()

            #保存模型
            best_prec = min(prec, best_prec)
            torch.save({'epoch':epoch+1,
                        'state_dict': model.state_dict(),
                        'best_prec': best_prec,
                        'scheduler': lr_scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()},args.models_save)
            if best_prec >= prec:
                shutil.copyfile(
                    args.models_save,
                    "checkpoint_best.pth"
                )
    else:
        output_log("=>train or eval:'{}'".format(args.train_flag), logging)
        if os.path.isfile("checkpoint_best.pth"):
            #恢复训练最好的模型
            output_log("=> loading checkpoint '{}'".format('./checkpoint_best.pth'),
                    logging)
            checkpoint = torch.load("./checkpoint_best.pth")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            output_log("=> loaded checkpoint '{}' "
                        .format('./checkpoint_best.pth'), logging)
            
            #测试网络效果
            accuracy, percision, recall, F1 = eval_net(dataloader, model)

            #输出最终统计结果
            output_log("=>accuracy:'{0}', percision:'{1}', recall:'{2}', F1:'{3}'".format(accuracy, percision, recall, F1), logging)
        else:
            output_log("=>train or eval:'{}', error-> no checkpoint".format('eval'), logging)


if __name__ == "__main__":
    main()  

