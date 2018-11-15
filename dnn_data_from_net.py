import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# X IV個數
x_col = 4

# Y 目標欄位總類
y_classify = 3

# 讀取來自 datasets 的 iris 資料集
iris = datasets.load_iris()

X = iris.data
Y = iris.target

# 將目標欄位轉換成數字代號
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(input_dim=x_col, units=y_classify))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1), metrics=['accuracy'])

# 檢視模型架構
model.summary()

# 訓練模型
model.fit(X, dummy_y, epochs=200, batch_size=10)

# 測試模型
loss, accuracy = model.evaluate(X, dummy_y)

# 輸出測試結果
print('Loss: %.2f%%' % (loss*100))
print('Accuracy: %.2f%%' % (accuracy*100))
