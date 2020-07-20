import gensim
import numpy as np
import os
import torch
from torch.utils.data import Dataset,DataLoader

def build_word2id(file):
    """
    :param file: word2id保存地址
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./Dataset/train.txt', './Dataset/validation.txt', './Dataset/test.txt']
    print(path)
    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    with open(file, 'w', encoding='utf-8') as f:
        for w in word2id:
            f.write(w+'\t')
            f.write(str(word2id[w]))
            f.write('\n')
    return word2id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    print("vec_len:{}".format(len(word_vecs[0])))
    return word_vecs


#数据读取主要使用torch.utils.data.DataLoader，其需要Dataset类型
#torch.utils.data.Dataset是抽象类，通过继承Dataset类并重写__len__与__getitem__方法实现自定义数据读取
#继承Dataset，重写关键函数。
class text_cnn_Data(Dataset):
    def __init__(self, word_vecs, word2id, data_path,transform=None):
        #[[0, word1, word2..],[1, word1, word2..]...]
        self.processed_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                if len(sp) == 0:
                    continue
                self.processed_data.append(sp)

        self.word2vec = word_vecs
        self.word2id = word2id
        self.seq_len = 666

    def __len__(self):
        return int(len(self.processed_data)-1)

    def __getitem__(self, idx):
        #_PAD_
        #len = 50
        label = int(self.processed_data[idx][0])
        train_data = []
        for word in self.processed_data[idx][1:]:
            word_id = self.word2id[word]
            train_data.append(word_id)
            #train_data.append(self.word2vec[word_id])
            #train_data.extend(self.word2vec[word_id])
            if len(train_data) == self.seq_len:
                break
        for tmp in range(self.seq_len - len(train_data)):
            train_data.append(self.word2id['_PAD_'])
            #train_data.extend(self.word2vec[self.word2id['_PAD_']])
        #print(label)
        return np.array(train_data), np.array(label)


if __name__ == "__main__":
    word2id = build_word2id('./Dataset/word2id.txt')
    word_vecs = build_word2vec("./Dataset/wiki_word2vec_50.bin", word2id, "word2vecs.txt")
    print("Process completed")

    dataloader = DataLoader(text_cnn_Data(word_vecs, word2id, './Dataset/test.txt'),
                         batch_size=1,
                         shuffle=True,
                         num_workers=2)
    
    for i, data in enumerate(dataloader):
        print(data[0].shape)
        print(data[1])
        print("------------")
        if i > 0:
            break
