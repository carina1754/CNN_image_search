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

# 데이터 학습
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

# 테스트 데이터로 평가해보기
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))


# 새로운 데이터로 테스트
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

name = []
test = []
name.append('input_data')
test.append(Dataization('./temp_data/input_data.jpg'))

# 테스트 데이터로 예측결과 보기
test = np.array(test)
model = load_model('img_model.h5')
predict = np.argmax(model.predict(test), axis=-1)
#predict = model.predict_classes(test)

# 예측 결과 출력
res_categorie = str(categories[predict[0]])
print(name[0] + " - Predict : "+ res_categorie)

file_num =0
# 입력 이미지 데이터를 데이터셋의 예측된 카테고리에 저장
res_categorie = str(categories[predict[0]])
for root, dir, file in os.walk(groups_folder_path+res_categorie+'\\'):
    file_num = len(file)
new_train = cv2.imread('./temp_data/input_data.jpg')
cv2.imwrite(groups_folder_path+res_categorie+'\\'+res_categorie+str(file_num+1)+'.jpg', new_train)

img_temp = np.full((500,500,3),255,dtype=np.uint8)
cv2.imwrite('./temp_data/blank.jpg', img_temp)
img_temp = cv2.imread('./temp_data/blank.jpg')
cv2.putText(img_temp, res_categorie, (110,250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0))
cv2.imshow('predict result!', img_temp)

cv2.waitKey()
cv2.destroyAllWindows()