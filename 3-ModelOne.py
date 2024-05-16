import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

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

for i in df.columns:
    sb.displot(df[i])
    plt.show()