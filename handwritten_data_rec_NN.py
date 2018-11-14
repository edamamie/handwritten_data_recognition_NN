'''
    Handwritten Digit Recognition
    784-300-10 network structure
    inputs should be scaled to [0,1]
    sigmoid activation function for all hidden and output neurons
    learning rate = 0.01
    batch size = 100
    number of epochs for training = 1000
    for repeatability, use a random seed of 3520
'''
# Import libraries
import numpy as np
np.random.seed(3520)
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import mnist
from sklearn.metrics import confusion_matrix



# Load dataset
X = mnist.load_images('train-images-idx3-ubyte')
T = mnist.load_labels('train-labels-idx1-ubyte')
tX = mnist.load_images('t10k-images-idx3-ubyte')
tT = mnist.load_labels('t10k-labels-idx1-ubyte')
X = X.reshape(784,-1)
tX = tX.reshape(784,-1)
T = keras.utils.to_categorical(T, num_classes=10)
tT = keras.utils.to_categorical(tT, num_classes=10)
X = X.T
tX = tX.T

# Set/get relevant parameters
num_samples, num_inputs = X.shape
num_outputs = T.shape[1]
batch_size = 100
epochs = 1000
learning_rate = 0.01

# Create neural network
model = Sequential()

model.add(Dense(units=300, activation='sigmoid', input_dim=num_inputs))
#model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=num_outputs, activation='sigmoid'))

model.summary()
#stochastic gradient descent optimizer
sgd = keras.optimizers.SGD(lr=learning_rate)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

# Train neural network
history = model.fit(
    X, T,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(tX, tT),
    shuffle=False
)
y_pred = model.predict(X)
y_pred = np.argmax(y_pred, axis=1)
y_pred = np.reshape(y_pred, (-1,1))

y_train = np.argmax(T, axis=1)
y_train = np.reshape(y_train, (-1,1))

# Confusion Matrix
cm = confusion_matrix(y_train, y_pred)
print(cm)

score = model.evaluate(X, T, verbose=0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
