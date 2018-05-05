#coding:utf8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import RepeatVector, TimeDistributedDense, Activation
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
import time
import numpy as np
import re
from textProcessing import pad_sequences,tokenize,vectorize_stories,loadfile,chtokenize,loaddic,vectorize_x,jiebacut

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


    # vocab = sorted(reduce(lambda x, y: x | y, (set(tmp_list) for tmp_list in input_list + tar_list)))

    vocab = loaddic('./corpus/smalldic.txt')

    print ('-----------')
    # print vocab
    print ('-----------')
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1  # keras进行embedding的时候必须进行len(vocab)+1
    # input_maxlen = max(map(len, (x for x in input_list)))
    # tar_maxlen = max(map(len, (x for x in tar_list)))
    input_maxlen = 70
    tar_maxlen = 17
    output_dim = vocab_size
    hidden_dim = 100

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

    en_de_model.load_weights('en_de_weights1-40.h5')

    print('Compiling...')
    time_start = time.time()
    en_de_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    time_end = time.time()
    print('Compiled, cost time:%fsecond!' % (time_end - time_start))

    # # input_text = loadfile('./corpus/content-12147.txt')
    # input_text = loadfile('./corpus/content1-500.txt')
    #
    # input_list = []
    # for tmp_input in input_text:
    #     input_list.append(chtokenize(tmp_input))
    #
    # inputs_train = vectorize_x(input_list, word_to_idx, input_maxlen, tar_maxlen, vocab_size)
    #
    # out_predicts = en_de_model.predict(inputs_train)
    # for i_idx, out_predict in enumerate(out_predicts):
    #     predict_sequence = []
    #     tempstr = ''
    #     for predict_vector in out_predict:
    #         next_index = np.argmax(predict_vector)
    #         next_token = idx_to_word[next_index]
    #         # print next_token
    #         tempstr += next_token
    #         predict_sequence.append(next_token)
    #     print tempstr
    #     # print('Predict output:', predict_sequence)
    #
    # print ('Train Ended')

# def predict(input_text):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostbyname(socket.gethostname())
    port = 50008
    sock.bind((host,port))
    sock.listen(5)
    while True:
        conn,addr = sock.accept()
        data = conn.recv(1024)
        list = []
        # input_text = '实际上，上周主管部门就和大唐打过招呼了，内部消息人士透露，国资委已经就李小琳任职问题和大唐进行沟通，但李小琳本人至今未报到。情况比较复杂，上述人士表示，目前还不敢完全确定，不排除后续还有变化。'
        tmp = 'BEG '+data+' END'
        tmp = jiebacut(tmp)
        list.append(tmp)
        result = ''
        input_list = []
        for tmp_input in list:
            print (tmp_input)
            print ('---!--!---')
            input_list.append(chtokenize(tmp_input))
        inputs_train = vectorize_x(input_list, word_to_idx, input_maxlen)
        out_predicts = en_de_model.predict(inputs_train)
        for i_idx, out_predict in enumerate(out_predicts):
            predict_sequence = []
            tempstr = ''
            for predict_vector in out_predict:
                next_index = np.argmax(predict_vector)
                next_token = idx_to_word[next_index]
                # print next_token
                tempstr += next_token
                predict_sequence.append(next_token)
            print (tempstr)
            result = tempstr

            print('Predict output:', predict_sequence)
        reply = result
        conn.send(reply.encode())



if __name__ == '__main__':
    train()
