import pandas as pd
import time
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Dropout
from numpy import concatenate
from math import sqrt

def fit_model(train_x,train_y,test_x,test_y):
    model= Sequential()
    model.add(LSTM(256, input_shape=(train_x.shape[1], train_x.shape[2]),return_sequences=True))#input_shape(step,length)
    model.add(LSTM(128))
    model.add(Dense(768,activation='tanh',kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Dense(2,activation='tanh'))
    model.compile(loss='mae', optimizer='adam')
    #fit model
    history=model.fit(train_x, train_y, epochs=50, batch_size=10, validation_data=(test_x, test_y), verbose=2, shuffle=False)
    #save model
    model_path="D:\\Machine Learning\\Pyprojects\\NN\\paper3\\paper_paras.h5"
    model.save(model_path)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def train_set(dataf):
    train_index= 80
    test_index= 40
    data=dataf.values
    train = data[:train_index, :]
    test = data[train_index:train_index+test_index, :]
    train_x, train_y = train[:, 0:1536], train[:, 1536:]
    test_x, test_y = test[:, 0:1536], test[:, 1536:]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x,train_y,test_x,test_y

def fit_trans(raw_data):
    values=raw_data.values
    values= values.astype('float32')
    values[:, 0:1536]= values[:, 0:1536]*1000
    #encoder= LabelEncoder()
    #values_train= encoder.fit_transform(train_I)
    return pd.DataFrame(values)

def load_data():
    pathfile= "D:\\Machine Learning\\Pyprojects\\NN\\paper3\\data.csv"
    ori_data= pd.read_csv(pathfile, header=None)
    value= ori_data.values
    return ori_data, value #type(ori_data, values)= (pandas.DataFrame, nddrary)

if __name__ == '__main__':
    data, value=load_data()
    print(data.describe())
    reframed = fit_trans(data)
    train_X,train_Y,test_X,test_Y=train_set(reframed)
    fit_model(train_X,train_Y,test_X,test_Y)