import numpy as np # Matrix and vector computation package
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import np_utils

import scipy.io as sio

x_train1 = sio.loadmat('../db/speech_cmd_trainbatch.mat')
y_train1 = sio.loadmat('../db/speech_cmd_trainbatchlabels.mat')
data1 = x_train1.get('batch') 
label1 = y_train1.get('batchlabels') 
x_train = np.array(data1) # For converting to numpy array
y_train = np.array(label1) # For converting to numpy array
y_train=y_train-1
x_test1 = sio.loadmat('../db/speech_cmd_testbatch.mat')
y_test1 = sio.loadmat('../db/speech_cmd_testbatchlabels.mat')
data1 = x_test1.get('testbatch') 
label1 = y_test1.get('testbatchlabels') 
x_test = np.array(data1) # For converting to numpy array
y_test = np.array(label1) # For converting to numpy array
y_test = y_test-1

batch = 100
epoch=50
hidden_units = 200
classes = 35

img=x_train
lab=y_train
img_test=x_test
lab_test=y_test
lab = np_utils.to_categorical(lab, classes)
lab_test = np_utils.to_categorical(lab_test, classes)

img=img[0:63400,0:60,0:39]
lab=lab[0:63400,0:35]
img_test=img_test[0:42300,0:60,0:39]
lab_test=lab_test[0:42300,0:35]

print(img.shape[1:])
a=np.zeros((100,4))
a[1][1]=1
import time
for i in range(1):

    model = Sequential()
    model.add(LSTM(hidden_units,input_shape =img.shape[1:], batch_size = batch))
    model.add(Dense(classes))
    
    model.add(Activation('softmax'))    
    model.compile(optimizer = 'RMSprop', loss='mean_squared_error', metrics = ['accuracy'])

    start_time = time.time() 
#    model.fit(img, lab, batch_size = batch,epochs=epoch,verbose=1)
#    history=model.fit(X_train, y_train, epochs=10,  validation_split = 0.1, batch_size = 400, verbose = 1, shuffle = 1)
    model.fit(img, lab,
          batch_size=batch,
          epochs=epoch,
          verbose=1,
          validation_data=(img_test, lab_test))

    
#    scores = model.evaluate(img_test, lab_test, batch_size=batch)
#    predictions = model.predict(img_test, batch_size = batch)
#    print('LSTM test score:', scores[0])
#    print('LSTM test accuracy:', scores[1])
#    a[i][0]=scores[0]
#    a[i][1]=scores[1]
#    a[i][2]=time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))