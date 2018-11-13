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
csv = "iris.csv"

# X IV個數
x_col = 4

# Y 目標欄位總類
y_classify = 3

# CV 交叉驗證次數
cross=10

# 1. 請刪除其餘欄位
# 2. 最右側欄位為目標欄位
# 3. 請刪除表頭

dataframe = pandas.read_csv(csv, header=None)
dataset = dataframe.values

X = dataset[:, 0:x_col].astype(float)
Y = dataset[:, x_col]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)

# 隨機分類 80%訓練, 20%測試
train_x, test_x, train_t, test_t = train_test_split(
    X, dummy_y, train_size=0.8, test_size=0.2)

model = Sequential()
model.add(Dense(input_dim=x_col, units=y_classify))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1), metrics=['accuracy'])

# 訓練模型
model.fit(train_x, train_t, epochs=100, batch_size=10)

# 查看可輸出的參數
# print(model.metrics_names)

# 測試模型
loss, accuracy = model.evaluate(test_x, test_t)

# 輸出測試結果
print('Loss: %.2f%%' % (loss*100))
print('Accuracy: %.2f%%' % (accuracy*100))

# 預測模型
# Y = model.predict_classes(test_x)

# 預測結果比對
# _, T_index = np.where(test_t > 0)
# print()
# print('Predict Result')
# print(Y == T_index)

# 切割資料，使用測試與訓練資料。
