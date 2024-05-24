import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras import layers
from keras import models


# Helper method for identifying the top crop in a prediction.
def identify_top_crops(predictions, crop):
    top_indices = np.argsort(predictions)[::-1][:2]
    top_crops = [(crop[i], predictions[i]) for i in top_indices]
    return top_crops


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


dfCrop = pd.read_csv("Data/Crop_Recommendation.csv")

# Mapping the crops to a number
crops = np.unique(dfCrop['Crop'])
crop_dictionary = dict(zip(crops, range(len(crops))))
dfCrop['Crop'] = dfCrop['Crop'].map(crop_dictionary)

# Making the model
X = dfCrop.drop("Crop", axis=1)
y = dfCrop['Crop']

model = models.Sequential()
model.add(layers.Dense(14, activation='relu'))
model.add(layers.Dense(7, activation='relu'))
model.add(layers.Dense(len(crops), activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.2, epochs=20, batch_size=200)

acc_chart(history)
loss_chart(history)

# Test Data
crop_names = list(crop_dictionary.keys())
# Chose high nutrient levels and high humidity resembling conditions for rice.
test_data_rice = np.array([[100, 30, 20, 25, 70, 6.5, 200]])
# Selected moderate nutrient levels, temperature, and humidity resembling conditions for mango.
test_data_mango = np.array([[50, 25, 30, 30, 60, 6.0, 150]])
# Opted for lower nutrient levels, higher temperature, and moderate humidity resembling conditions to orange.
test_data_orange = np.array([[20, 10, 15, 35, 50, 5.5, 100]])
# test_data_coffee = np.array([[60, 25, 20, 25, 65, 6.0, 150]])
# test_data_lentil = np.array([[70, 25, 30, 25, 65, 6.0, 180]])

predictions_rice = model.predict(test_data_rice)
predictions_mango = model.predict(test_data_mango)
predictions_orange = model.predict(test_data_orange)
# predictions_coffee = model.predict(test_data_coffee)
# predictions_lentil = model.predict(test_data_lentil)

top_crops_rice = identify_top_crops(predictions_rice[0], crop_names)
top_crops_mango = identify_top_crops(predictions_mango[0], crop_names)
top_crops_orange = identify_top_crops(predictions_orange[0], crop_names)
# top_crops_coffee = identify_top_crops(predictions_coffee[0], crop_names)
# top_crops_lentil = identify_top_crops(predictions_lentil[0], crop_names)

print("Test dataset 1 - Conditions suitable for Rice:")
print("Top crops:", top_crops_rice)

print("\nTest dataset 2 - Conditions suitable for Mango:")
print("Top crops:", top_crops_mango)

print("\nTest dataset 3 - Conditions suitable for Orange:")
print("Top crops:", top_crops_orange)
# print("Test dataset 4 - Conditions suitable for Coffee:")
# print("Top crops:", top_crops_coffee)
# print("\nTest dataset 5 - Conditions suitable for Lentil:")
# print("Top crops:", top_crops_lentil)
