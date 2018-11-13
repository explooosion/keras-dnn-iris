import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

numpy.random.seed(0)

# 檔案名稱
csv = "iris.csv"

# X IV個數
x_col = 4

# Y 目標欄位總類
y_classify = 3

# CV 交叉驗證次數
cross = 10

dataframe = pandas.read_csv(csv, header=None)
dataset = dataframe.values

X = dataset[:, 0:x_col].astype(float)
Y = dataset[:, x_col]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
    model = Sequential()
    model.add(Dense(input_dim=x_col, units=y_classify))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.1), metrics=['accuracy'])

    return model


# 定義分類器的訓練模式
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10)

# N次交叉驗證用途
# KFold 基本版, 隨機切割
# kfold = KFold(n_splits=cross, shuffle=True, random_state=0)

# StratifiedKFold 平均切割, 避免資料區間不一致
# n_splits 切割堆疊數量
# shuffle 將資料亂數排序
# random_state 亂數種子
kfold = StratifiedKFold(n_splits=cross, shuffle=True, random_state=None)

# 執行 CV 
results = cross_val_score(estimator, X, Y, cv=kfold, scoring='accuracy')

print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# 使用交叉驗證，10次。
