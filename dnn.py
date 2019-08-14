import keras
import pandas
import sklearn
from sklearn.preprocessing import LabelEncoder

# 檔案名稱
csv = "iris_header.csv"

# IV 個數
x_col = 4

# DV 目標欄位總類
y_classify = 3

# 利用 pandas 工具讀取 csv 檔案
dataframe = pandas.read_csv(csv, header=1)
# dataframe = pandas.read_csv(csv, header=None)

# 取出 csv 資料集中的值，原本包含檔案大小等資訊
dataset = dataframe.values

# 從0開始取出 x_col=4 的欄位，型別為浮點數
X = dataset[:, 0:x_col].astype(float)
# 直接取出第 x_col 的欄位，型別根據資料來源(字串)
Y = dataset[:, x_col]

# 標籤轉碼器，將目標欄位轉換成數字代號
# 建構一個空的標籤轉換器
encoder = LabelEncoder()
# 將 DV 放入轉換器
encoder.fit(Y)
# 將資料標準化(normalized)，並存成 encoded_Y
encoded_Y = encoder.transform(Y)
# [0 1 0 2 2 0 2 1] 0~2:類別

# 資料型態轉換工具組，將目標欄位轉換成二維矩陣
# 由於 encoded_Y 是一個陣列 [0, 1, 1, 2, …]，模型必須將資料轉換成二維矩陣。
# 轉換後資料： [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1] … 根據Y種類決定矩陣長度
dummy_y = keras.utils.np_utils.to_categorical(encoded_Y)

# 建立模型，從keras.models提取需要建立空的模型
model = keras.models.Sequential()
# imput_dim 輸入層(IV), units 神經單元個數(DV)
model.add(keras.layers.Dense(input_dim=x_col, units=y_classify))
# Activation = 激勵函數(解決非線性問題)
model.add(keras.layers.Activation('softmax'))
# loss 損失函數(估量預測結果差異程度)
# opt 優化器，用於最佳解問題, prediction = wx + b
# SGD = 隨機梯度下降優化器(stochastic gradient decent)
# lr 學習率(Learning Rate)
# metrics 以 acc 作為評估指標
# https://keras-cn.readthedocs.io/en/latest/other/optimizers/
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])

# 檢視模型架構
model.summary()

# 訓練模型 epochs 迭代次數, batch_size 每次迭代(Iteration)的筆數
model.fit(X, dummy_y, epochs=200, batch_size=10)

# 測試模型
loss, accuracy = model.evaluate(X, dummy_y)

# 輸出測試結果
print('Loss: %.2f%%' % (loss*100))
print('Accuracy: %.2f%%' % (accuracy*100))