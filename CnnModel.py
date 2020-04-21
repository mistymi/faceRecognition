#CnnModel.py
#casia有目标数据的人脸模型
#获取文件夹下所有图片，根据人脸位置剪裁出人脸并保存
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow


from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

CASE_PATH = "D:/cuda/miniconda/envs/tf1.x/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
RAW_IMAGE_DIR = 'C:/Users/Administrator/Desktop/article/CASIA/'
DATASET_DIR = 'C:/Users/Administrator/Desktop/article/CASIA_GRAY/'

face_cascade = cv2.CascadeClassifier(CASE_PATH)

#存放数据到文件夹
def save_faces(img, name,x, y, width, height):
    image = img[y:y+height, x:x+width]
    cv2.imwrite(name, image)


image_list = os.listdir(RAW_IMAGE_DIR)#列出文件夹下所有的目录与文件
count = 0
for image_path in image_list:
    image = cv2.imread(RAW_IMAGE_DIR + image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.15,
                                          minNeighbors=5,
                                          minSize=(5, 5),
                                         )
    #cv2.imwrite('%ss%d.bmp' % (DATASET_DIR, count), gray)
    for(x, y, width, height) in faces:
        save_faces(gray, '%ss%d.bmp' % (DATASET_DIR, count), x, y - 30, width, height+30)
    count = count + 1

# 尺寸变换
def resize_without_deformation(image, size=(100, 100)):
    height, width, _ = image.shape
    longest_edge = max(height, width)
    top, bottom, left, right = 0, 0, 0, 0
    if height < longest_edge:
        height_diff = longest_edge - height
        top = int(height_diff / 2)
        bottom = height_diff - top
    elif width < longest_edge:
        width_diff = longest_edge - width
        left = int(width_diff / 2)
        right = width_diff - left

    image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    resized_image = cv2.resize(image_with_border, size)

    return resized_image

# 读取图片，返回重置大小后的图片及标签
def read_image(size=None):
    data_x, data_y = [], []
    for i in range(1, 200):
        try:
            im = cv2.imread('C:/Users/Administrator/Desktop/article/CASIA_GRAY/s%s.bmp' % str(i))
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if size is None:
                size = (100, 100)
            im = resize_without_deformation(im, size)
            data_x.append(np.asarray(im, dtype=np.int8))
            data_y.append(str(int((i - 1) / 11.0)))
        except IOError as e:
            print(e)
        except:
            print('INEXIT')

    return data_x, data_y

#读入图像及标签
IMAGE_SIZE = 100
raw_images, raw_labels = read_image(size=(IMAGE_SIZE, IMAGE_SIZE))
raw_images, raw_labels = np.asarray(raw_images, dtype = np.float32),  np.asarray(raw_labels, dtype = np.int32) # 把图像转换为float类型，方便归一化

ont_hot_labels = np_utils.to_categorical(raw_labels)#on-hot编码

#按照7:3划分训练集和测试集，并打乱顺序
train_input, valid_input, train_output, valid_output =train_test_split(raw_images,
                                                                       ont_hot_labels,
                                                                       test_size = 0.3)

#数据归一化
train_input /= 255
valid_input /= 255

#搭建模型
face_recognition_model = Sequential()
face_recognition_model.add(Conv2D(32, (3, 3), padding='valid',
                                  strides=(1, 1),
                                  dim_ordering='tf',
                                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                  activation= 'relu'))


face_recognition_model.add(MaxPool2D(pool_size=(2, 2)))
face_recognition_model.add(Dropout(0.2))

face_recognition_model.add(Conv2D(64, (3, 3), padding='valid',
                                  strides=(1, 1),
                                  dim_ordering='tf',
                                  activation= 'relu'))



face_recognition_model.add(MaxPool2D(pool_size=(2, 2)))
face_recognition_model.add(Dropout(0.2))

face_recognition_model.add(Flatten())
face_recognition_model.add(Dense(512, activation='relu'))#全连接层
face_recognition_model.add(Dropout(0.4))
face_recognition_model.add(Dense(len(ont_hot_labels[0]), activation='sigmoid'))
face_recognition_model.add(Activation('softmax'))#输出分类结果


face_recognition_model.summary()


learning_rate = 0.01
decay = 1e-6
momentum = 0.8
nesterov = True
#sgd作为反向传播优化器
sgd_optimizer = SGD(lr=learning_rate, decay=decay,
                    momentum=momentum, nesterov=nesterov)
#编译模型，交叉熵作为损失函数
face_recognition_model.compile(loss='categorical_crossentropy',
                               optimizer=sgd_optimizer,
                               metrics=['accuracy'])

#开始训练
batch_size = 20 #每批训练数据量的大小
epochs = 100#训练100次
face_recognition_model.fit(train_input, train_output,
                           epochs=epochs,
                           batch_size=batch_size,
                           shuffle=True,
                           validation_data=(valid_input, valid_output))

#训练完成，在测试集上评估并保存模型
print(face_recognition_model.evaluate(valid_input, valid_output, verbose=0))
MODEL_PATH='face_model.h7'
face_recognition_model.save(MODEL_PATH)