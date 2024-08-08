import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

##온도기록도 데이터셋만 학습시킨 모델입니다

current_directory = os.getcwd()
print("Current Directory:", current_directory)

cg_folder = './Diabetes_dateset/flip_temper/train/CG/'
dm_folder = './Diabetes_dateset/flip_temper/train/DM/'

example_file = './Diabetes_dateset/example_data/CG045_R.csv'
example_filename = os.path.basename(example_file)


cg_data = []
for filename in os.listdir(cg_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(cg_folder, filename)
        df = pd.read_csv(filepath)
        # 행열 크기를 맞추기 위해 0으로 채워진 배열 생성
        pad_array = np.zeros((200, 100))
        # 기존 데이터를 새로운 배열에 복사
        pad_array[:df.shape[0], :df.shape[1]] = df.values
        cg_data.append(pad_array)


dm_data = []
for filename in os.listdir(dm_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(dm_folder, filename)
        df = pd.read_csv(filepath)
        pad_array = np.zeros((200, 100))
        pad_array[:df.shape[0], :df.shape[1]] = df.values
        dm_data.append(pad_array)


X = np.concatenate((np.array(cg_data), np.array(dm_data)), axis=0)
y = np.concatenate((np.zeros(len(cg_data)), np.ones(len(dm_data))), axis=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#CNN 모델
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 100, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))


#####예측 파일 처리
df = pd.read_csv(example_file)
pad_array = np.zeros((200, 100))
pad_array[:df.shape[0], :df.shape[1]] = df.values

input_data = pad_array.reshape(1, 200, 100, 1)  #CNN
# input_data = pad_array.reshape(1, 100, 200)  #RNN


prediction = model.predict(input_data)

a = model.evaluate(X_train, y_train)[1]
b = model.evaluate(X_test, y_test)[1]
# 모델 평가 출력
print('\nCNN 온도기록도 학습 모델 train 정확도: ', "{:.2f}".format(a))
print('CNN 온도기록도 학습 모델 test 정확도: ', "{:.2f}".format(b))

# 예측 결과 출력
if prediction[0] < 0.5:
    print('\n', filename, '에 대한 예측 결과: CG')
else:
    print('\n',filename, '에 대한 예측 결과: DM')

'''#이 코드는 RogisticRegression()으로 돌렸을 때의 코드입니다.
current_directory = os.getcwd()
print("Current Directory:", current_directory)

cg_folder = './Diabetes_dateset/flip_temper/train/CG/'
dm_folder = './Diabetes_dateset/flip_temper/train/DM/'


example_file = './Diabetes_dateset/example_data/CG045_R.csv'
example_filename = os.path.basename(example_file)


cg_data = []
for filename in os.listdir(cg_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(cg_folder, filename)
        df = pd.read_csv(filepath)
        pad_array = np.zeros((200, 100))
        pad_array[:df.shape[0], :df.shape[1]] = df.values
        cg_data.append(pad_array)


dm_data = []
for filename in os.listdir(dm_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(dm_folder, filename)
        df = pd.read_csv(filepath)
        pad_array = np.zeros((200, 100))
        pad_array[:df.shape[0], :df.shape[1]] = df.values
        dm_data.append(pad_array)


X = np.concatenate((np.array(cg_data), np.array(dm_data)), axis=0)
y = np.concatenate((np.zeros(len(cg_data)), np.ones(len(dm_data))), axis=0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LogisticRegression()

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)


model.fit(X_train_flatten, y_train)

train_accuracy = model.score(X_train_flatten, y_train)
test_accuracy = model.score(X_test_flatten, y_test)


print(model, '온도기록도 학습 모델 train 정확도: {:.2f}'.format(train_accuracy))
print(model, '온도기록도 학습 모델 test 정확도: {:.2f}'.format(test_accuracy))

#####예측 파일 처리
df = pd.read_csv(example_file)
pad_array = np.zeros((200, 100))
pad_array[:df.shape[0], :df.shape[1]] = df.values

input_data = pad_array.reshape(1, -1)


prediction = model.predict(input_data)


if prediction[0] == 1:
    print('\n{}에 대한 예측 결과: CG'.format(example_filename))
else:
    print('\n{}에 대한 예측 결과: DM'.format(example_filename))
'''
