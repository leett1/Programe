# author - Richard Liao
# Dec 26 2016
import numpy as np
#import pandas as pd
#import pickle
#from collections import defaultdict
#import re

#from bs4 import BeautifulSoup
#from gensim.models import Word2Vec
#import sys
#import os
#import random
#from keras.preprocessing.text import Tokenizer, text_to_word_sequence
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical

#from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
#from nltk import tokenize
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
#import keras.preprocessing.text as T
#from keras.preprocessing.text import Tokenizer
from config import manager
from keras.losses import binary_crossentropy
train_context = manager("train")
test_context = manager("test")

def data_generator():
    for data in train_context.document.load():
        yield data["vector"], np.squeeze(data["label"], axis=1)

def get_data():
    for data in test_context.document.load(never_stop=False):
        yield [data["vector"], np.squeeze(data["label"], axis=1)]

def test_data():
    return [[data["vector"], np.squeeze(data["label"], axis=1)] for data in test_context.document.load_all()]

class AttLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            print(ait,mask)
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output
    def get_config(self):
        """这个方法用于保存构造函数参数，如果没有这个方法载入模型的时候会报错"""
        config = {'attention_dim':self.attention_dim}
        base_config = super(AttLayer,self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

'''embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True,
                            mask_zero=True)
'''

if __name__ == "__main__":
    sentence_input_ids = Input(shape=(250, ), dtype='int32')
    sentence_input_vectors = train_context.embedding.keras_embedding_layer(sentence_input_ids)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(sentence_input_vectors)
    l_att = AttLayer(100)(l_lstm)
    sentEncoder = Model(sentence_input_ids, l_att)

    review_input = Input(shape=(10, 250), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)

    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(100)(l_lstm_sent)
    preds = Dense(1, activation='sigmoid')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss=binary_crossentropy,
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - Hierachical attention network")
    '''model.fit(data_generator(), validation_data=load_files(batch_size,path1),
              nb_epoch=10, batch_size=50)
              '''
    model.fit_generator(data_generator(), epochs=6, steps_per_epoch=manager("train").document.num_batches)
    #score = model.evaluate_generator(get_data(), steps=int(3577 / manager("train").document.num_batches))
    model.save('two_attention_mode190317.h5')

    """
    def HAN():
        sentence_input = Input(shape=(max_features,), dtype='int32')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
        l_att = AttLayer(100)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)
        review_input = Input(shape=(maxlen, max_features), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(LSTM(100, return_sequences=True))(review_encoder)
        l_att_sent = AttLayer(100)(l_lstm_sent)
        preds = Dense(1, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)
    """
