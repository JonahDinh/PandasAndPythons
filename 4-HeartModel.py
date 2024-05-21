import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras import layers
from keras import models


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


def graph_stuff(dfCopy):
    # We want to look at the numbers for the participants
    dfCopy['condition'] = dfCopy['condition'].map({0: "Healthy", 1: "Heart Patient"})
    dfCopy['sex'] = dfCopy['sex'].map({0: "Male", 1: "Female"})

    # Let's look at the male graphs for this
    dfMale = dfCopy[dfCopy['sex'] == 'Male']
    sb.countplot(x='condition', data=dfMale)
    plt.title("Male Stats")
    plt.show()

    dfFemale = dfCopy[dfCopy['sex'] == 'Female']
    sb.countplot(x='condition', data=dfFemale)
    plt.title("Female Stats")
    plt.show()

    condHealth = dfCopy['condition'] == 'Healthy'
    condSick = dfCopy['condition'] == 'Heart Patient'

    # compare age with health
    plt.hist(dfCopy[condHealth]['age'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfCopy[condSick]['age'], color='g', alpha=0.5, bins=15, label='Heart Problems')
    plt.legend()
    plt.title("Health Count vs Age")
    plt.show()

    # compare choelestoral with health
    plt.hist(dfCopy[condHealth]['chol'], color='r', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfCopy[condSick]['chol'], color='y', alpha=0.5, bins=15, label='Heart Problems')
    plt.legend()
    plt.title("Health Count vs Choelestoral")
    plt.show()

    # compare thalach or the maximum achived heart rate with health
    plt.hist(dfCopy[condHealth]['thalach'], color='r', alpha=0.5, bins=15, label="Healthy")
    plt.hist(dfCopy[condSick]['thalach'], color='y', alpha=0.5, bins=15, label='Heart Problems')
    plt.legend()
    plt.title("Health Count vs thalach")
    plt.show()


dfHeart = pd.read_csv("Data/heart_cleveland_upload.csv")

print(dfHeart.head().to_string())
print(dfHeart.describe())

# We want to graph som of the fields
graph_stuff(dfHeart.copy())

# identify the outliers

df_high_col = dfHeart[dfHeart['chol'] > 500]
print(df_high_col.to_string())
dfHeart = dfHeart[dfHeart['chol'] < 500]

df_low_heart = dfHeart[dfHeart['thalach'] < 80]
print(df_low_heart.to_string())
dfHeart = dfHeart[dfHeart['thalach'] > 80]

X = dfHeart.drop("condition", axis=1)
y = dfHeart['condition']

print("Shape of x is %s " % str(X.shape))
print("Shape of y is %s " % str(y.shape))

model = models.Sequential()

model.add(layers.Dense(13, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.2, epochs=200)

# If we don't specify a batch size, the default is 32
# Since we have 295 entries, this means that we will run
# 10 different samples of size 32 for each epoch

acc_chart(history)
loss_chart(history)

X_at_risk = np.array([[62, 1, 3, 145, 250, 1, 2, 120, 0, 1.4, 1, 1, 0]], dtype=np.float64)
y_at_risk = (model.predict(X_at_risk) > 0.5).astype(int)
print(y_at_risk[0])

X_healty = np.array([[50, 1, 2, 129, 196, 0, 0, 163, 0, 0, 0, 0, 0]], dtype=np.float64)
y_healthy = (model.predict(X_healty) > 0.5).astype(int)
print(y_healthy[0])
