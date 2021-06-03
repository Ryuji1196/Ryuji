from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

#モデルの保存

#データのロード
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#データの整形
X_train = X_train.reshape([-1,28,28,1])
X_test = X_test.reshape([-1,28,28,1])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#モデルの定義
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_data=(X_test, y_test))

#精度の評価
scores = model.evaluate(X_test,y_test, verbose=1)
print("test loss:", scores[0])
print("test accuracy:", scores[1])

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape((28,28)), 'gray')
plt.suptitle("10 images of test data",fontsize=20)
plt.show()

# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
print(pred)

model.summary()

#resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# 重みを保存
model.save(os.path.join(result_dir, 'model.h5'))

files.download( '/content/results/model.h5' ) 
