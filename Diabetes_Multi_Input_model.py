import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import natsort

##주석은 최종 모델인 다중입력모델 파일에만 적었습니다

##또 다양하고 반복적인 데이터 가공으로 인해서 데이터셋 파일이 여러 폴더에 중복되어 들어있습니다.
##가공하지 않은 원본 데이터가 들어있는 파일은 Control_Group, DM Group이고 모든 train과 test 데이터셋은 임의대로 나눴습니다(CG 41~45, DM 111~122)
##모든 가공을 거친 최종 데이터셋은 flip_image와 flip_temper 폴더에 있습니다!
##마지막으로 파이썬 파일을 가끔 Diabetes_dataset에 만들기도 했기에 경로가 잘 못 되는 경우가 발생할 수 있습니다.

#예측시켜 볼 예제 데이터의 경로
example_csv_path = './Diabetes_dateset/example_data/CG045_R.csv'
example_csv_filename = os.path.basename(example_csv_path)
example_image_path = './Diabetes_dateset/example_data/CG045_M_R.png'
example_image_filename = os.path.basename(example_image_path)


#CSV 데이터 경로(예제 데이터는 위 경로로 이미 옮겨두었기에 입력 데이터에 포함되지 않습니다)
train_cg_csv_path = './Diabetes_dateset/flip_temper/train/CG/'
train_dm_csv_path = './Diabetes_dateset/flip_temper/train/DM/'
test_cg_csv_path = './Diabetes_dateset/flip_temper/test/CG/'
test_dm_csv_path = './Diabetes_dateset/flip_temper/test/DM/'

#이미지 데이터 경로(예제 데이터는 위 경로로 이미 옮겨두었기에 입력 데이터에 포함되지 않습니다)
train_cg_images_folder = './Diabetes_dateset/flip_image/train/CG/'
train_dm_images_folder = './Diabetes_dateset/flip_image/train/DM/'
test_cg_images_folder = './Diabetes_dateset/flip_image/test/CG/'
test_dm_images_folder = './Diabetes_dateset/flip_image/test/DM/'

#CSV 데이터 전처리(csv파일의 flip 처리 과정은 따로 실행시켜 flip_temper폴더에 저장한 후 이 파일에서는 불러오기만 했기에 해당 소스코드엔 flipe 처리 과정이 생략되었습니다)
csv_data = []
labels = []
for csv_path, label in [(train_cg_csv_path, 0), (train_dm_csv_path, 1)]:
    #CG = 0, DM = 1로 레이블 지정
    for filename in natsort.natsorted(os.listdir(csv_path)):
        if filename.endswith('.csv'):
            filepath = os.path.join(csv_path, filename)
            df = pd.read_csv(filepath)
            #print(filename, '의 원본 shape: ', df.shape)
            pad_array = np.zeros((200, 100))#모두 0값인 200x100 크기의 배열 생성
            #csv파일의 행열의 크기가 각각 다 달라 원본 csv를 pad_array에 데이터를 padding해 전처리
            pad_array[:df.shape[0], :df.shape[1]] = df.values
            #print(filename, '의 전처리 후 shape: ', pad_array.shape, '\n')
            csv_data.append(pad_array)
            labels.append(label)

X_csv = np.array(csv_data)
y_csv = np.array(labels)

#이미지 데이터 전처리(마찬가지로 이미지 데이터의 flip 처리와 resizing 처리 과정 또한 따로 실행시켜 flip_image에 저장했기에 해당 소스코드엔 생략되었습니다)
image_data = []
image_labels = []
for image_folder, label in [(train_cg_images_folder, 0), (train_dm_images_folder, 1)]:
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            filepath = os.path.join(image_folder, filename)
            image = tf.keras.preprocessing.image.load_img(filepath, target_size=(88, 200))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image_data.append(image)
            image_labels.append(label)

X_images = np.array(image_data)
y_images = np.array(image_labels)

#데이터셋 분할
X_csv_train, X_csv_test, y_csv_train, y_csv_test = train_test_split(X_csv, y_csv, test_size=0.2, random_state=42)
X_images_train, X_images_test, y_images_train, y_images_test = train_test_split(X_images, y_images, test_size=0.2, random_state=42)

print('Number of CSV files in train_cg_csv_path:', len(os.listdir(train_cg_csv_path)))
print('Number of CSV files in train_dm_csv_path:', len(os.listdir(train_dm_csv_path)))
print('Number of image files in train_cg_images_folder:', len(os.listdir(train_cg_images_folder)))
print('Number of image files in train_dm_images_folder:', len(os.listdir(train_dm_images_folder)))

print('X_csv shape:', X_csv.shape)
print('y_csv shape:', y_csv.shape)
print('X_images shape:', X_images.shape)
print('y_images shape:', y_images.shape)

#CSV 입력 모델
csv_input = Input(shape=(200, 100, 1))
csv_conv = Conv2D(32, kernel_size=(3, 3), activation='relu')(csv_input)
csv_pool = MaxPooling2D(pool_size=(2, 2))(csv_conv)
csv_flatten = Flatten()(csv_pool)
csv_dense = Dense(128, activation='relu')(csv_flatten)

#이미지 입력 모델
image_input = Input(shape=(88, 200, 3))
image_conv1 = Conv2D(16, (3, 3), activation='relu')(image_input)
image_pool1 = MaxPooling2D(2, 2)(image_conv1)
image_conv2 = Conv2D(32, (3, 3), activation='relu')(image_pool1)
image_pool2 = MaxPooling2D(2, 2)(image_conv2)
image_flatten = Flatten()(image_pool2)
image_dense = Dense(128, activation='relu')(image_flatten)

#CSV와 이미지, 두 개의 입력을 결합
combined = concatenate([csv_dense, image_dense])
combined_dense = Dense(64, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(combined_dense)

#다중 입력 모델
model = Model(inputs=[csv_input, image_input], outputs=output)

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#원본 이미지 1개당 10개의 변형된 이미지 생성으로 데이터량 증강
datagen = ImageDataGenerator(rescale=1./255, #이미지 픽셀 값을 0과 1 사이로 정규화
                                   width_shift_range=0.05, #좌우 이동
                                   rotation_range=3, #회전
                                   fill_mode='nearest')

train_generator = datagen.flow(X_images_train, y_images_train, batch_size=10)


model.fit([X_csv_train, X_images_train], y_csv_train, epochs=10, batch_size=32)

print('\n다중 입력 모델 train 정확도: ', "{:.2f}".format(a))
print('다중 입력 모델 test 정확도: ', "{:.2f}".format(b))


#############모델 학습/생성 끝, 여기부턴 예측하기#############
#똑같이 인풋 데이터 처리 과정 거친 후 예측

example_csv_data = []
example_csv_labels = []
example_df = pd.read_csv(example_csv_path)
pad_array = np.zeros((200, 100))
pad_array[:example_df.shape[0], :example_df.shape[1]] = example_df.values
example_csv_data.append(pad_array)
example_csv_labels.append(0)  # 예제 데이터의 라벨은 0으로 가정합니다.

X_example_csv = np.array(example_csv_data)
y_example_csv = np.array(example_csv_labels)


example_image_data = []
example_image_labels = []
example_image = tf.keras.preprocessing.image.load_img(example_image_path, target_size=(88, 200))
example_image = tf.keras.preprocessing.image.img_to_array(example_image)
example_image_data.append(example_image)
example_image_labels.append(0)  # 예제 데이터의 라벨은 0으로 가정합니다.

X_example_images = np.array(example_image_data)
y_example_images = np.array(example_image_labels)


#예제 데이터(CG)에 대하여 예측
predictions = model.predict([X_example_csv, X_example_images])
predicted_label = 'CG' if predictions[0] < 0.5 else 'DM'

a = model.evaluate([X_csv_train, X_images_train], y_csv_train)[1]
b = model.evaluate([X_csv_test, X_images_test], y_csv_test)[1]

print('\n', example_csv_filename,'&',example_image_filename, ' 에 대한 예측 결과: ', predicted_label)

#ImageDatagenerator로 변형된 이미지 2열 5행으로 시각화
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

for i in range(2):
    for j in range(5):
        augmented_image, _ = train_generator.next()
        axs[i, j].imshow(augmented_image[0], interpolation='bicubic' , aspect=2.3) #aspect 세로를 가로의 2.3배로 보여줌
        axs[i, j].axis('off')

plt.tight_layout()
plt.show()




