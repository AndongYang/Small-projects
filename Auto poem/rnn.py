import argparse
import logging
import time
import os
import math
import shutil
import datetime
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

from rnn_dataloader import Rnn_Data

#初始化参数
parser = argparse.ArgumentParser(description='rnn')

parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_num', default=10, type=int, metavar='N',
                    help='epoch number')
parser.add_argument('--batch_size', default=8, type=int, metavar='BS',
                    help='batch size of training')
parser.add_argument('--start_words', default="雨余芳草净沙尘",
                    type=str, metavar='PATH',
                    help='init poems')
parser.add_argument('--train', default="train", type=str, metavar='te',
                    help='train or eval')

#我的笔记本上gpu0是核显，1是独显
parser.add_argument('--gpu', default='1', type=str, help='GPU id to use.')
parser.add_argument('--train_data_dir', default="./tang.npz",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--max_gen_len', default=69, type=int, metavar='mgl',
                    help='lenth of res')
parser.add_argument('--embedding_dim', default=128, type=int, metavar='ed',
                    help='embedding_dim of LSTM')
parser.add_argument('--hidden_dim', default=256, type=int, metavar='hd',
                    help='hidden_dim of LSTM')
parser.add_argument('--num_layers', default=1, type=int, metavar='hd',
                    help='num_layers of LSTM')
parser.add_argument('--models_save', default='./checkpoint.pth', type=str, 
                    metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

#获得参数
global args
args = parser.parse_args()
#初始化日志
logging.basicConfig(filename="./rnn.log", level=logging.ERROR)
#指定使用的显卡
GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#GLOBAL_DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICE']=args.gpu

#记录开始时间
start_time = time.time()

class Net(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim

        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, args.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim,512)
        self.fc2 = nn.Linear(512,1024)
        self.fc3 = nn.Linear(1024,vocab_size)


    def forward(self, inputs, hidden=None):
        embeds = self.embeddings(inputs)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = inputs.size()
        if hidden is None:
            h_0 = inputs.data.new(args.num_layers*1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = inputs.data.new(args.num_layers*1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden

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

def train(dataloader, word2ix, model, loss_cal, optimizer, epoch):
    '''Returen avg_loss'''
    model.train()

    for i, data in enumerate(dataloader):
        inputs, labels = data[0].to(GLOBAL_DEVICE), data[1].to(GLOBAL_DEVICE)
        labels = labels.view(-1)
        
        optimizer.zero_grad()
        outputs, hidden = model(inputs)
        loss = loss_cal(outputs, labels)
        loss.backward()
        optimizer.step()

        #每隔一定轮数输出一次结果
        avg_loss = avg_loss_log.update(loss.item())
        if i % args.print_freq == 0 or i == len(dataloader):
            output_log("=> " 
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4f}\t'
                    .format(epoch, i, len(dataloader), batch_time=time.time() - start_time, 
                        loss=avg_loss), logging)

    return avg_loss



def generate(model, start_words, ix2word, word2ix):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input_words = torch.Tensor([word2ix['<START>']]).view(1, 1).long().cuda()
    hidden =  torch.zeros((2, args.num_layers , 1, args.hidden_dim), dtype=torch.float).cuda()
    model.eval()
    with torch.no_grad():
        for i in range(args.max_gen_len):
            output, hidden = model(input_words, hidden)
            # 如果在给定的句首中，input_words为句首中的下一个字
            if i < start_words_len:
                w = results[i]
                input_words = input_words.data.new([word2ix[w]]).view(1, 1)
            # 否则将output作为下一个input_words进行
            else:
                top_index = output.data[0].topk(1)[1][0].item()
                w = ix2word[top_index]
                results.append(w)
                input_words = input_words.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
                del results[-1]
                break
        return results


def output_log(data, logger=None):
    print("{}:{}".format(datetime.datetime.now(), data))
    if logger is not None:
        logger.critical("{}:{}".format(datetime.datetime.now(), data))


def get_data():
    datas = np.load(args.train_data_dir)
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    dataloader = DataLoader(Rnn_Data(args.train_data_dir),
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=2)
    return dataloader, ix2word, word2ix


def main():
    #获取训练数据
    train_dataloader, ix2word, word2ix = get_data()
    output_log("=>loaded train_data.", logging)

    # tmp = 0
    # for i ,data in enumerate(train_dataloader):
    #     print("{}:{}".format(i, data))
    #     tmp+=1
    #     if tmp ==10:
    #         break
    # return

    #初始化网络模型
    model = Net(len(word2ix)).to(GLOBAL_DEVICE)
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

    if args.train == 'train':
        output_log("=>train or eval:'{}'".format('train'), logging)
        #记录最准确的模型，方便最后输出参数
        best_prec = math.inf

        #开始主循环
        for epoch in range(args.epoch_num):
            #训练网络
            prec = train(train_dataloader, word2ix, model, loss_cal, optimizer, epoch)
            #更新lr
            lr_scheduler.step()

            #使用网络生成结果
            res = generate(model, args.start_words, ix2word, word2ix)
            res = "".join(res)

            #输出结果
            output_log("=>epoch{0}:{1}".format(epoch+1, res),logging)

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
        output_log("=>train or eval:'{}'".format('eval'), logging)
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
            
            #使用网络生成结果
            res = generate(model, args.start_words, ix2word, word2ix)
            res = " ".join(i for i in res)

            #输出结果
            output_log("=>res:{0}".format(res), logging)
        else:
            output_log("=>train or eval:'{}', error-> no checkpoint".format('eval'), logging)
    

if __name__ == "__main__":
    main()    

