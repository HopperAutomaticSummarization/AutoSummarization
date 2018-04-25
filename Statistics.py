#coding:utf8

import numpy as np
from textProcessing import tokenize,vectorize_stories,loadfile,chtokenize
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

input_text = loadfile('./corpus/content-12147.txt')
tar_text = loadfile('./corpus/title-12147.txt')
# input_text = loadfile('./corpus/content1-500.txt')
# tar_text = loadfile('./corpus/title1-500.txt')
dic = {}
input_list = []
tar_list = []
input_i =0
tar_i =0
for tmp_input in input_text:
    result = chtokenize(tmp_input)
    # for r in result:
    #     r = r.strip()
    #     if(dic.has_key(r)):
    #         dic[r] +=1
    #     else:
    #         dic[r] =1
    input_list.append(result)
    print input_i
    input_i += 1
for tmp_tar in tar_text:
    result = chtokenize(tmp_tar)
    # for r in result:
    #     r = r.strip()
    #     if(dic.has_key(r)):
    #         dic[r] +=1
    #     else:
    #         dic[r] =1
    tar_list.append(result)
    print tar_i
    tar_i += 1

# dicfile = open('./corpus/dic.txt','a')
# for d in dic:
#     dicfile.write(d+'\n')
# dicfile.close()

# vocab = sorted(reduce(lambda x, y: x | y, (set(tmp_list) for tmp_list in input_list + tar_list)))
vocab = []

dicfile = open('./corpus/dic.txt','r')
line = dicfile.readline()
while line:
    vocab.append(line.strip().decode("utf8"))
    line = dicfile.readline()
dicfile.close()
print vocab
print vocab[0]

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1  # keras进行embedding的时候必须进行len(vocab)+1
input_maxlen = max(map(len, (x for x in input_list)))
tar_maxlen = max(map(len, (x for x in tar_list)))
# output_dim = vocab_size
hidden_dim = 50

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Input max length:', input_maxlen, 'words')
print('Target max length:', tar_maxlen, 'words')
print('Dimension of hidden vectors:', hidden_dim)
print('Number of training stories:', len(input_list))
print('Number of test stories:', len(input_list))

# word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # 编码时需要将字符映射成数字index
# for word in word_to_idx:
#     print word,word_to_idx[word]
# idx_to_word = dict((i + 1, c) for i, c in enumerate(vocab))  # 解码时需要将数字index映射成字符
# for idx in idx_to_word:
#     print idx,idx_to_word[idx]