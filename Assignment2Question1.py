import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras import layers
from keras import models


# Methods taken from class to display loss and accuracy metrics
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


dfLoan = pd.read_csv("Data/loan.csv")
# The Occupation Field is not needed for this (lots of Different occupations).
dfLoan = dfLoan.drop("occupation", axis=1)

# The Categorized Strings should be converted to integer Vales for the analysis.
dfLoan['gender'] = dfLoan['gender'].map({"Male": 0, "Female": 1})
dfLoan['marital_status'] = dfLoan['marital_status'].map({"Single": 0, "Married": 1})
dfLoan['loan_status'] = dfLoan['loan_status'].map({"Denied": 0, "Approved": 1})

# The education level should be broken up into 3 different types.
dfLoan['education_level'] = dfLoan['education_level'].map(
    {"High School": 0, "Bachelor's": 1, "Associate's": 1, "Master's": 2, "Doctoral": 2, })

# Create a Heatmap for the given DataFrame.
sb.heatmap(dfLoan.corr(), annot=True)
plt.show()

# Create Histograms that compare:
condApproved = dfLoan['loan_status'] == 1
condDenied = dfLoan['loan_status'] == 0

# compare age against approved/denied
# 15 bins for age range
plt.hist(dfLoan[condApproved]['age'], color='g', alpha=0.5, bins=15, label="Approved")
plt.hist(dfLoan[condDenied]['age'], color='r', alpha=0.5, bins=15, label='Denied')
plt.legend()
plt.title("Age against Approved/Denied")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Compare education_level against approved/denied
# 3 Bins for 3 different education levels
plt.hist(dfLoan[condApproved]['education_level'], color='g', alpha=0.5, bins=3, label="Approved")
plt.hist(dfLoan[condDenied]['education_level'], color='r', alpha=0.5, bins=3, label='Denied')
plt.legend()
plt.title("Education Level against Approved/Denied")
plt.xlabel("Education Level")
plt.ylabel("Frequency")
plt.show()

# Compare marital_status against approved/denied
# 2 Bins for 2 marital statuses
plt.hist(dfLoan[condApproved]['marital_status'], color='g', alpha=0.5, bins=2, label="Approved")
plt.hist(dfLoan[condDenied]['marital_status'], color='r', alpha=0.5, bins=2, label='Denied')
plt.legend()
plt.title("Marital Status against Approved/Denied")
plt.xlabel("Marital Status")
plt.ylabel("Frequency")
plt.show()

# Noticed an outlier with a high salary, decided to remove it
dfLoan = dfLoan[dfLoan['income'] < 150000]

# Create an appropriate mode given the Modified DataFrame you have prepared.
X = dfLoan.drop("loan_status", axis=1)
y = dfLoan['loan_status']

print("Shape of x is %s " % str(X.shape))
print("Shape of y is %s " % str(y.shape))

# As part of this you should try different combinations of Layers, Loss Functions, and optimizers.
model = models.Sequential()

# model.add(layers.Dense(24, activation='relu'))
# model.add(layers.Dense(18, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=20)
# history = model.fit(X, y, validation_split=0.2, epochs=100)
# history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=10)
# history = model.fit(X, y, validation_split=0.2, epochs=150, batch_size=20)

# Display the loss and accuracy
acc_chart(history)
loss_chart(history)

# Test data, this should be approved. (40Y Married Male, High Educated, 100k and 800 Credit score)
X_approved = np.array([[40, 0, 2, 1, 100000, 800]], dtype=np.float64)
y_approved = (model.predict(X_approved) > 0.5).astype(int)
print(y_approved[0])

# Test data, this should be denied. (25Y Single Female, Low Educated, 40k and 600 Credit score)
X_denied = np.array([[25, 1, 0, 0, 40000, 600]], dtype=np.float64)
y_denied = (model.predict(X_denied) > 0.5).astype(int)
print(y_denied[0])

# Additional Test Data
# X_middle = np.array([[29, 1, 1, 0, 55000, 700]], dtype=np.float64)
# y_middle = (model.predict(X_middle) > 0.5).astype(int)
# print(y_middle[0])
#
# X_mike = np.array([[28, 0, 0, 50000, 0, 500]], dtype=np.float64)
# y_myself = (model.predict(X_myself) > 0.5).astype(int)
# print(y_myself[0])

# Saving the model
model.save("Models/loan.keras")

# Additional Graphs for More Information - Used this data for better understanding in creating test data
#   Men, Salary above 60k and Credit score over 600 are all the general loan approval numbers

# plt.hist(dfLoan[condApproved]['gender'], color='g', alpha=0.5, bins=2, label="Approved")
# plt.hist(dfLoan[condDenied]['gender'], color='r', alpha=0.5, bins=2, label='Denied')
# plt.legend()
# plt.title("Gender Approved/Denied")
# plt.xlabel("Gender")
# plt.ylabel("Frequency")
# plt.show()
# plt.hist(dfLoan[condApproved]['income'], color='g', alpha=0.5, bins=15, label="Approved")
# plt.hist(dfLoan[condDenied]['income'], color='r', alpha=0.5, bins=15, label='Denied')
# plt.legend()
# plt.title("Income against Approved/Denied")
# plt.xlabel("Income")
# plt.ylabel("Frequency")
# plt.show()
# plt.hist(dfLoan[condApproved]['credit_score'], color='g', alpha=0.5, bins=5, label="Approved")
# plt.hist(dfLoan[condDenied]['credit_score'], color='r', alpha=0.5, bins=5, label='Denied')
# plt.legend()
# plt.title("Credit Score against Approved/Denied")
# plt.xlabel("Credit Score")
# plt.ylabel("Frequency")
# plt.show()
