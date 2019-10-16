# coding=utf-8
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from gensim.models import Word2Vec
import re
import random

class Sentences():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        file_list = os.listdir(self.path)
        for file_index in file_list:
            a = []
            for line in open('F:\Lee\论文\我的项目\CNN_word2wec_sentence\word_train/' + str(file_index), 'rb'):
                line = re.sub(r'[^\x00-\x7F]+',' ', str(line))
                # char = ['\t', '\n', '\\t', '\\n', '<?php', '?>', ';', '<html', '</html>', '<body', '</body>', '>',
                #         '<head', '</head>', '<meta', '/>', '<title', '</title>', '<style>', '</style>', '<script',
                #         '</script>', '<a', '</a>', '<p', '</p>', '<div', '</div>', '<b', '</b>']
                char = ['\t', '\n']
                for i in char:
                    line = line.strip(i)
                line = line.split(' ')
                a = a + line
            yield a



#训练词向量模型
path = "F:\Lee\论文\我的项目\CNN_word2wec_sentence\word_train/"
sentences = Sentences(path)
word_model = Word2Vec(sentences,size=128,window=10,min_count=5)
output = "word_train190313.model"
word_model.save(output)

