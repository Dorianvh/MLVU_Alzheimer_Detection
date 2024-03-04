from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
import pandas as pd
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


x_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
model = create_model()
model = compile_model(model)
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=1/6)
