import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
import glob
import cv2
import numpy as np
import os
import os.path
import natsort
from PIL import Image
import warnings
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
warnings.filterwarnings('ignore')

##이미지 데이터셋만 학습시킨 모델입니다

example_image_path = './Diabetes_dateset/example_data/CG045_M_R.png'
example_filename = os.path.basename(example_image_path)

train_dir = './Diabetes_dateset/flip_image/train'
test_dir = './Diabetes_dateset/flip_image/test'

train_cg_dir = './Diabetes_dateset/flip_image/train/CG'
train_dm_dir = './Diabetes_dateset/flip_image/train/DM'

test_cg_dir = './Diabetes_dateset/flip_image/test/CG'
test_cg_filenames = os.listdir(test_cg_dir)
test_dm_dir = './Diabetes_dateset/flip_image/test/DM'
test_dm_filenames = os.listdir(test_dm_dir)

dic_cgdm_filenames = {}
dic_cgdm_filenames['CG'] = test_cg_filenames
dic_cgdm_filenames['DM'] = test_dm_filenames

train_cg_fnames = os.listdir(train_cg_dir)
train_dm_fnames = os.listdir(train_dm_dir)
print('Total train CG images: ', len(os.listdir(train_cg_dir)))
print('Total train DM images: ', len(os.listdir(train_dm_dir)))
print('Total test CG images: ', len(os.listdir(test_cg_dir)))
print('Total test DM images: ', len(os.listdir(test_dm_dir)))


train_datagen = ImageDataGenerator(rescale=1./255, #이미지 픽셀 0과 1 사이로 정규화
                                   width_shift_range=0.05, #좌우 이동
                                   rotation_range=3, #회전
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255, #이미지 픽셀 0과 1 사이로 정규화
                                   width_shift_range=0.05, #좌우 이동
                                   rotation_range=3, #회전
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=16,
                                                    color_mode='rgb',
                                                    class_mode='binary',
                                                    target_size=(88,200))

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=4,
                                                  color_mode='rgb',
                                                  class_mode='binary',
                                                  target_size=(88,200))

print(train_generator.class_indices)#CG는 0, DM은 1

model = tf.keras.models.Sequential()

model.add(Conv2D(16, (3,3), activation='relu', input_shape=(88,200,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch=4,
                              epochs=50,
                              validation_steps=4,
                              verbose=2)


example_image_data = []
example_image_labels = []
example_image = tf.keras.preprocessing.image.load_img(example_image_path, target_size=(88, 200))
example_image = tf.keras.preprocessing.image.img_to_array(example_image)
example_image_data.append(example_image)
example_image_labels.append(0)

X_example_images = np.array(example_image_data)

predictions = model.predict(X_example_images)
predicted_label = 'CG' if predictions[0] < 0.5 else 'DM'

a = model.evaluate(train_generator)[1]
b = model.evaluate(test_generator)[1]

print('\n이미지 학습 모델 train 정확도:' , "{:.2f}".format(a))
print('이미지 학습 모델 test 정확도: ', "{:.2f}".format(b))
print('\n', example_filename, '에 대한 예측 결과: ', predicted_label)
