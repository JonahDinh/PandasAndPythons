import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras import layers
from keras import models


def do_visual(dfSample):
    sb.heatmap(dfSample.corr(), annot=True)
    plt.show()


def acc_chart(results):
    plt.title("Accuracy of model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def loss_chart(results):
    plt.title("Model losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


dfWine = pd.read_csv("Data/winequality-red.csv")

# print(dfWine.head().to_string())
# print(dfWine.describe())
do_visual(dfWine)

X = dfWine.drop("quality", axis=1)
y = dfWine['quality']

model = models.Sequential()
model.add(layers.Dense(11, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.2, epochs=150, batch_size=100)

acc_chart(history)
loss_chart(history)

md_wine = np.array([[7.5, 0.71, 0, 1.8, 0.76, 10.88, 34.2, 0.00, 3.5, 0.55, 9.5]], dtype=np.float64)
y_pred = model.predict(md_wine)
print(y_pred)

# We can save the model to a file which can then be read
model.save("Models/wine.keras")
