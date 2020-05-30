from keras.datasets import mnist
dataset=mnist.load_data('mymnist.db')
len(dataset)
train , test = dataset
X_train , y_train = train
X_train.shape
X_test , y_test = test
X_test.shape
ima1_label= y_train[0]
img1_label
ima1d=img1.reshape(28*28)
img1d.shape
X_train.shape
X_train_1d=X.train.reshape(-1, 28*28)
X_train = X_train_1d.astype('float32')
y_train.shape
from keras.utils.np_utils import to_categorical
y_train_cat
from kera.model import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=512, input_dim = 28*28, actuvation = 'relu'))
i=0
for i in range():
    model.add(Dense(units=128, activation = 'relu'))

model.add(Dense(units=10, activation = 'softmax'))
model.summary()
from keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(), loss = 'categorical_crossentropy', metrics = ['accuracy'] )
h = model.fit(X_train, y_train, epocs = 8)
X_test_1d = X_test.reshape(-1, 28*28)
X_test= X_train_1d.astype('float32')
y_test_cat= to_categorical(y_test)
model.predict(X_test)
p=h.history['accuracy']
h.history['accuracy'][7]
with open('file.txt', 'w') as f:
    f.write(str(p[7]))
model.save('mymodel.h1')
