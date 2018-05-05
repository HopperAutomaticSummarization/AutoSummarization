#coding:utf8
import numpy as np
import re,jieba
import string
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# 分词
def jiebacut(text):
    text = text.strip()
    seg_list = jieba.cut(text)
    result = " ".join(seg_list)
    return result

# 载入文件到list
def loadfile(filepath):
    result = []
    file = open(filepath,'r')
    line = file.readline()
    while line:
        tmp = line.strip()
        tmp = 'BEG '+tmp+' END'
        # tmp = line.replace(' ','')
        tmp = jiebacut(tmp)
        # print tmp
        result.append(tmp)
        line = file.readline()
    file.close()
    return result

# 填充序列
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# 词转词向量-训练
def vectorize_stories(input_list, tar_list, word_idx, input_maxlen, tar_maxlen, vocab_size):
    x_set = []
    Y = np.zeros((len(tar_list), tar_maxlen, vocab_size), dtype=np.bool)
    # print 'Done'
    for _sent in input_list:
        # print _sent
        # print _sent[0]
        # x = [word_idx[w.encode('utf8')] for w in _sent]
        x = []
        for w in _sent:
            if(word_idx.has_key(w.encode('utf8'))):
                x.append(word_idx[w.encode('utf8')])
            else:
                pass
                # print w
        # print x
        x_set.append(x)
    for s_index, tar_tmp in enumerate(tar_list):
        for t_index, token in enumerate(tar_tmp):
            if(word_idx.has_key(token.encode('utf8'))):
                Y[s_index, t_index, word_idx[token.encode('utf8')]] = 1
            else:
                pass
                # print token

    return pad_sequences(x_set, maxlen=input_maxlen), Y

# 词转词向量-预测
def vectorize_x(input_list, word_idx, input_maxlen,):
    x_set = []
    # print 'Done'
    for _sent in input_list:
        # print _sent
        # print _sent[0]
        # x = [word_idx[w.encode('utf8')] for w in _sent]
        x = []
        for w in _sent:
            if(word_idx.has_key(w.encode('utf8'))):
                x.append(word_idx[w.encode('utf8')])
            else:
                pass
                # print w
        # print x
        x_set.append(x)
    return pad_sequences(x_set, maxlen=input_maxlen)

# 英文-词转成列表数据结构
def tokenize(sent):
    result = [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    return result

# 中文-词转成列表数据结构
def chtokenize(sent):
    tmp = sent.split(' ')
    result = []
    for t in tmp:
        if (t.strip() == ''):
            continue
        else:
            result.append(t)

    return result

# 生成字典
def generateDic(titleFile,contentFile,dicFile):
    dic={}
    frtitle = open(titleFile,'r')
    frcontent = open(contentFile, 'r')
    fwdic = open(dicFile,'a')
    line = frtitle.readline()
    while line:
        result = jiebacut(line)
        temp = result.split(" ")
        for t in temp:
            if(dic.has_key(t)):
                dic[t] +=1
            else:
                dic[t] =1
        line = frtitle.readline()

    line = frcontent.readline()
    while line:
        result = jiebacut(line)
        temp = result.split(" ")
        for t in temp:
            if(dic.has_key(t)):
                dic[t] +=1
            else:
                dic[t] =1
        line = frcontent.readline()

    for d in dic:
        # print d[0]+' '+str(d[1])
        if(d.strip()==''):
            continue
        fwdic.write(d+'\n')

    frtitle.close()
    frcontent.close()
    fwdic.close()
    print "Finished!"

# 载入字典
def loaddic(dicpath):
    file = open(dicpath,'r')
    dic = []
    line = file.readline()
    while line:
        dic.append(line.strip())
        line = file.readline()
    file.close()
    return dic

# 获取最大序列长度
def gettargetlen(input_textPath,tar_textPath):
    # input_text = loadfile('./corpus/content1-500.txt')
    # tar_text = loadfile('./corpus/title1-500.txt')
    input_text = loadfile(input_textPath)
    tar_text = loadfile(tar_textPath)
    input_maxlen = 0
    tar_maxlen = 0
    for tmp_input in input_text:
        length = len(chtokenize(tmp_input))
        if(length > input_maxlen):
            input_maxlen = length
    for tmp_tar in tar_text:
        length = len(chtokenize(tmp_tar))
        if(length > tar_maxlen):
            tar_maxlen = length

    print input_maxlen
    print tar_maxlen



if __name__ == '__main__':
    pass
    gettargetlen('./corpus/content-12147.txt','./corpus/title-12147.txt')
    generateDic('./corpus/title-12147.txt','./corpus/content-12147.txt','./corpus/dic.txt')
    # chtokenize(loadfile('./corpus/title1-500.txt')[0])

