import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 檔案名稱
# csv = "iris.csv"

# X IV個數
x_col = 4

# Y 目標欄位總類
y_classify = 3

# 1. 請刪除其餘欄位
# 2. 最右側欄位為目標欄位
# 3. 請刪除表頭

np.random.seed(0)

iris = datasets.load_iris()

X = iris.data
Y = iris.target

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# to_categorical converts the numbered labels into a one-hot vector
dummy_y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(input_dim=x_col, units=y_classify))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1), metrics=['accuracy'])

# 訓練模型
model.fit(X, dummy_y, epochs=200, batch_size=10)

# 查看可輸出的參數
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
