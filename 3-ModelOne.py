import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers


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


df = pd.read_csv("Data/kc_house_data.csv")

# print(df.head().to_string())
#
# print("\n\n Shape and Size")
# print(df.shape)
#
# print(df["price"].describe)
#
# print(df.dtypes)

# Create a new field called reg_year and take that
# To be the first 4 characters of the date
df['reg_year'] = df['date'].str[:4]
df['reg_year'] = df['reg_year'].astype('int')

# print (df.head().to_string())
# print(df.dtypes)

# we want to add a new series called house_age to the DataFrame
# If the house is renovated, the age will be the difference between the reg_year and the build year
# The house age will be the difference between the reg_year and the rennovation year
df['house_age'] = np.NAN

for i, j in enumerate(df['yr_renovated']):
    if j == 0:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_built']
    else:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_renovated']

# We want to get rid of the unecessary fields

df.drop(['yr_built', 'date', 'yr_renovated', 'reg_year'], axis=1, inplace=True)
df.drop(['id', 'zipcode', 'lat', 'long'], axis=1, inplace=True)
# print(df.head().to_string())

# Normally we would have to do individual check for bad data values
# This would consist of going through each of the series to see if there was bad data there

# Proving there is bad data
# df_bad = df[df['house_age'] < 0]
# print(df_bad.head().to_string())

df = df[df['house_age'] >= 0]

# for i in df.columns:
#     sb.displot(df[i])
#     plt.show()


# sb.pairplot(df['price', 'bedroom'])
# plt.show()

# The heat map can be used to show correlation between different series
# plt.figure(figsize=(30, 20))
# sb.heatmap(df.corr(), annot=True)
# plt.show()

# We want to be able to predict the price of a house based upon
# other series values. So we are going to create a dataframe for our input values
# and another datafram for our output values, both of these become part of our model
# andalysis later on.

X = df.drop("price", axis=1)
Y = df['price']

print(X.head().to_string())
print(Y.head().to_string())

print(X.shape)
print(Y.shape)

# Define our model
# We are going to have one input layer, one hidden layer and 1 output layer.

my_model = keras.Sequential()
my_model.add(layers.Dense(14, activation="relu"))
my_model.add(layers.Dense(4, activation="relu"))
my_model.add(layers.Dense(1))

# Compile our model & put these into a results

my_model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
results = my_model.fit(X, Y, validation_split=0.33, batch_size=64, epochs=10)
