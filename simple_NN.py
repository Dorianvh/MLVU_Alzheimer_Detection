from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #Prevents a plotting error not sure why

def create_model(input_shape = (600,)):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape))  # first dense layer, 32 hidden units
    model.add(Activation('relu'))  # activation layer
    model.add(Dense(3))  # second dense layer
    model.add(Activation('softmax'))  # output class probabilities
    model.summary()
    return model

def compile_model(model, learning_rate = 0.001):
    optimizer = Adam(learning_rate=learning_rate)  # lr is the learning rate
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, plot = False):
    history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=1 / 6)

    if plot == True:
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel(' Loss (categorical crossentropy)')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()




x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
model = create_model()
model = compile_model(model)
train_model(model,True)