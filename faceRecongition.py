#h9人脸识别
import cv2
import numpy as np
import keras
from keras.models import load_model

CASE_PATH = "D:/cuda/miniconda/envs/tf1.x/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASE_PATH)
#加载模型
face_recognition_model = keras.Sequential()
MODEL_PATH= 'myface_model.h5'
face_recognition_model = load_model(MODEL_PATH)

# 尺寸变换
def resize_without_deformation(image, size=(64, 64)):
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




cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 利用分类器识别出哪个区域为人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(30, 30))
    # 截取脸部图像，重置大小后提交给模型
    for (x, y, width, height) in faces:
        img = image[y:y + height, x:x + width]
        img = resize_without_deformation(img)

        img = img.reshape((1, 64, 64, 3))
        img = np.asarray(img, dtype=np.float32)
        img /= 255.0

        result = face_recognition_model.predict_classes(img)
        print("result", result)

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if result[0] == 8:
            cv2.putText(image, 'LiuJie', (x, y - 2), font, 0.7, (0, 255, 0), 2)
        elif result[0] == 9:
            cv2.putText(image, 'HuangDongMei', (x, y - 2), font, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'UnKnow', (x, y - 2), font, 0.7, (0, 255, 0))

    cv2.imshow('', image)
    cv2.waitKey(0)