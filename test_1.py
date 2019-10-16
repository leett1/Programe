import numpy as np
from config import manager
import keras
import time
from keras.layers import Lambda
from keras import backend
import matplotlib.pyplot as plt
from train_model import AttLayer
train_context = manager("train")
test_context = manager("test")


#onattention
from keras.layers import  Lambda
from keras import backend

def get_data():
    for data in test_context.document.load(never_stop=False):
        yield [data["vector"]]

def get_y_label():
    data = test_context.document.load_all()
    y = np.squeeze(data["label"], axis=1)

    return y

def get_filename():
    data = test_context.document.load_all()
    filename = np.squeeze(data["filename"], axis=1)
    return filename

if __name__ == "__main__":
    start = time.clock()
    #model = keras.models.load_model('two_attention_mode190317.h5', custom_objects={"AttLayer": AttLayer})    #two_attention
    model = keras.models.load_model('one_attention_mode190427_danx.h5', custom_objects={"AttLayer": AttLayer, "backend": backend}) #one_attention
   # model =keras.models.load_model('rnn_mode190427_10.h5',custom_objects={"AttLayer": AttLayer, "backend": backend})
    score = model.predict_generator(get_data(), steps=manager("test").document.num_batches)
    #score2 = model2.predict_generator(get_data(), steps=manager("test").document.num_batches)
    rate = [score[i][0] for i in range(len(score))]
    y_pred = [(rate[i] >= 0.5)*1 for i in range(len(rate))]
    y_test = get_y_label()
    filename = get_filename()
    tmp1 = []
    for i in zip(y_test, y_pred, filename):
        if abs(i[0]-i[1])>0.5:
            tmp1.append(i)
    print(tmp1)
    print(y_pred)
    from sklearn import metrics
    print('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred))
    print('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
    print('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
    print('F1-score: %.4f' % metrics.f1_score(y_test,y_pred))
    print('Precesion: %.4f' % metrics.precision_score(y_test,y_pred))
    metrics.confusion_matrix(y_test,y_pred)
    end = time.clock()
    print(str(end - start))