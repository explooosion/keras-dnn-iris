import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(0)

# 檔案名稱
csv = "iris_header.csv"

# IV 個數
x_col = 4

# DV 目標欄位總類
y_classify = 3

# 1. 請刪除其餘欄位
# 2. 最右側欄位為目標欄位
# 3. 請刪除表頭

dataframe = pandas.read_csv(csv, header=1)
# dataframe = pandas.read_csv(csv, header=None)
dataset = dataframe.values

X = dataset[:, 0:x_col].astype(float)
Y = dataset[:, x_col]

# 將目標欄位轉換成數字代號
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# 將目標欄位轉換成二維矩陣
dummy_y = np_utils.to_categorical(encoded_Y)

# 建立模型
model = Sequential()
# imput_dim 輸入層(IV), units 神經單元個數(DV)
model.add(Dense(input_dim=x_col, units=y_classify))
# 激活函數
model.add(Activation('softmax'))
# loss 損失函數, opt 隨機梯度下降優化器,
# lr 學習率, metrics 以 acc 作為評估指標
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1), metrics=['accuracy'])

# 訓練模型 epochs 迭代, batch_size 批量訓練
model.fit(X, dummy_y, epochs=200, batch_size=10)

# 查看可輸出的評估指標
# print(model.metrics_names)

# 測試模型
loss, accuracy = model.evaluate(X, dummy_y)

# 輸出測試結果
print('Loss: %.2f%%' % (loss*100))
print('Accuracy: %.2f%%' % (accuracy*100))

# 預測模型
# Y = model.predict_classes(X)

# 預測結果比對
# _, T_index = np.where(dummy_y > 0)
# print()
# print('Predict Result')
# print(Y == T_index)

# 未使用交叉驗證，未使用測試與訓練資料。
