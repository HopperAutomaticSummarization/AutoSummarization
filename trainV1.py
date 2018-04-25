# -*- encoding:utf-8 -*-

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import RepeatVector, TimeDistributedDense, Activation
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
import time
import numpy as np
from textProcessing import  tokenize,vectorize_stories,loadfile,chtokenize
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def train():
    # input_text = ['1 2 3 4 5'
    #               , '6 7 8 9 10'
    #               , '11 12 13 14 15'
    #               , '16 17 18 19 20'
    #               , '21 22 23 24 25']
    # tar_text = ['one two three four five'
    #             , 'six seven eight nine ten'
    #             , 'eleven twelve thirteen fourteen fifteen'
    #             , 'sixteen seventeen eighteen nineteen twenty'
    #             , 'twenty_one twenty_two twenty_three twenty_four twenty_five']

    # input_text = ['Hello this is Tom speaking Is that John ?'
    #               , 'Would you like to go swimming with me ?'
    #               , 'Ok see you then Bye'
    #               , 'Yeah I am free What time shall we meet ?'
    #               , 'How does it taste ?']
    # tar_text = ['Yes this is What s up ?'
    #             , 'That sounds great It s good weather for swimming  I d love to .'
    #             , 'See you'
    #             , 'At 3:00PM'
    #             , 'It tastes good you should try some.']

    #排序 生成数据格式为list的字典
    # vocab = sorted(reduce(lambda x, y: x | y, (set(tmp_list) for tmp_list in input_list + tar_list)))

    vocab = []
    dicfile = open('./corpus/dic.txt','r')
    line = dicfile.readline()
    while line:
        vocab.append(line.strip())
        line = dicfile.readline()
    dicfile.close()

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1  # keras进行embedding的时候必须进行len(vocab)+1
    # input_maxlen = max(map(len, (x for x in input_list)))
    # tar_maxlen = max(map(len, (x for x in tar_list)))
    input_maxlen = 99
    tar_maxlen = 22
    output_dim = vocab_size
    hidden_dim = 50

    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Input max length:', input_maxlen, 'words')
    print('Target max length:', tar_maxlen, 'words')
    print('Dimension of hidden vectors:', hidden_dim)
    # print('Number of training stories:', len(input_list))
    # print('Number of test stories:', len(input_list))
    print('-')
    print('Vectorizing the word sequences...')
    word_to_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # 编码时需要将字符映射成数字index
    idx_to_word = dict((i + 1, c) for i, c in enumerate(vocab))  # 解码时需要将数字index映射成字符


    decoder_mode = 3  # 0 最简单模式，1 [1]向后模式，2 [2] Peek模式，3 [3]Attention模式
    if decoder_mode == 3:
        encoder_top_layer = LSTM(hidden_dim, return_sequences=True)
    else:
        encoder_top_layer = LSTM(hidden_dim)

    if decoder_mode == 0:
        decoder_top_layer = LSTM(hidden_dim, return_sequences=True)
        decoder_top_layer.get_weights()
    elif decoder_mode == 1:
        decoder_top_layer = LSTMDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim
                                        , output_length=tar_maxlen, state_input=False, return_sequences=True)
    elif decoder_mode == 2:
        decoder_top_layer = LSTMDecoder2(hidden_dim=hidden_dim, output_dim=hidden_dim
                                         , output_length=tar_maxlen, state_input=False, return_sequences=True)
    elif decoder_mode == 3:
        decoder_top_layer = AttentionDecoder(hidden_dim=hidden_dim, output_dim=hidden_dim
                                             , output_length=tar_maxlen, state_input=False, return_sequences=True)

    en_de_model = Sequential()
    en_de_model.add(Embedding(input_dim=vocab_size,
                              output_dim=hidden_dim,
                              input_length=input_maxlen))
    en_de_model.add(encoder_top_layer)
    if decoder_mode == 0:
        en_de_model.add(RepeatVector(tar_maxlen))
    en_de_model.add(decoder_top_layer)

    en_de_model.add(TimeDistributedDense(output_dim))
    en_de_model.add(Activation('softmax'))
    print('Compiling...')
    time_start = time.time()
    en_de_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    time_end = time.time()
    print('Compiled, cost time:%fsecond!' % (time_end - time_start))

    for iter_num in range(5000):
        #构造数据 批量喂入
        input_text = loadfile('./corpus/content-12147.txt')
        tar_text = loadfile('./corpus/title-12147.txt')
        for i in range (0,58213):
            if (i == 58200):
                break
            input_list = []
            tar_list = []
            for tmp_input in input_text[i:i+20]:
                input_list.append(chtokenize(tmp_input,vocab))
            for tmp_tar in tar_text[i:i+20]:
                tar_list.append(chtokenize(tmp_tar,vocab))
            inputs_train, tars_train = vectorize_stories(input_list, tar_list, word_to_idx, input_maxlen, tar_maxlen, vocab_size)
            en_de_model.fit(inputs_train, tars_train, batch_size=4, nb_epoch=1, show_accuracy=True)
            i += 20

        # out_predicts = en_de_model.predict(inputs_train)
        # for i_idx, out_predict in enumerate(out_predicts):
        #     predict_sequence = []
        #     for predict_vector in out_predict:
        #         next_index = np.argmax(predict_vector)
        #         next_token = idx_to_word[next_index]
        #         predict_sequence.append(next_token)
        #     print('Target output:', tar_text[i_idx].decode('utf8'))
        #     print('Predict output:', predict_sequence)

        print('Current iter_num is:%d' % iter_num)
    en_de_model.save_weights('en_de_weights1-40.h5')
    print ('Train Ended')


if __name__ == '__main__':
    train()
