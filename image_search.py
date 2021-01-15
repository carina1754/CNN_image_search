import os, cv2, numpy as np
from sklearn.model_selection import train_test_split

groups_folder_path = './dataset-resized/'
categories = ["cardboard","glass","metal","paper","plastic","trash"]
num_classes = len(categories)

image_w = 28
image_h = 28

X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)] # 0으로 초기화된 1*(num_classes)배열
    print(label)
    print(idex)
    print(label[idex])
    label[idex] = 1
    img_dir = groups_folder_path + categorie + '\\'
    print(img_dir)

    for top, dir, f in os.walk(img_dir):
        for filename in f:
            print(img_dir+filename)
            img = cv2.imread(img_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/256)
            Y.append(label)
            print(img/256)
            print(label)
            
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
xy = (X_train, X_test, Y_train, Y_test)

np.save("./img_data.npy", xy)

from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
import cv2
 
X_train, X_test, Y_train, Y_test = np.load('./img_data.npy', allow_pickle=True)

model = Sequential()
model.add(Convolution2D(16, (3, 3), padding='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Convolution2D(64, (3, 3),  activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Convolution2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
  
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = 'softmax'))

# 모델 학습과정 설정
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
# 모델 학습
hist = model.fit(X_train, Y_train, batch_size=32, epochs=300)
 
model.save('img_model.h5')

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model
 
categories = ["cardboard","glass","metal","paper","plastic","trash"]
 
def Dataization(img_path):
    image_w = 28
    image_h = 28
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)
 
src = []
name = []
test = []
image_dir = "./dataset-resized_test/"
for file in os.listdir(image_dir):
    if (file.find('.jpg') != -1):      
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))
 
 
test = np.array(test)
model = load_model('img_model.h5')
predict = model.predict_classes(test)
 
for i in range(len(test)):
    print(name[i] + " : , Predict : "+ str(categories[predict[i]]))