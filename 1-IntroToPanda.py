import numpy as np
import pandas as pd

# create a panda series with index Titles
series1 = pd.Series([12, 14, 17, 19, 20], index=["one", "two", "three", "four", "five"])
print(series1)

series2 = pd.Series([33, 22, 11, 10])
print(series2)

workout = {"Mon": "Legs", "Tue": "Core", "Wed": "Biceps", "Thur": "Straight Fucking", "Fri": "Leg"}
sWorkout = pd.Series(workout)
print(sWorkout.loc["Mon"])

print(series2.loc[1])

df2 = pd.DataFrame(
    {
        "Numbers": pd.Series([11, 72, 23, 24]),
        "Names": pd.Series(["Kyrie", "Luka", "LeBron", "Kobe"]),
        "Status": pd.Categorical(["Alive", "Alive", "Alive", "Dead"]),
        "League": "NBA"
    }
)
positions = ["Shooting Guard", "Point Guard", "Small Forward", "Power Forward"]

df2['Positions'] = positions

# add a new Row
# 1 we can get the length (# rows) of the data frame by using the len( index property)
# we can then add at that location by just creating the appropriate row entry
df2.loc[len(df2.index)] = [30, "Steph", "Alive", "NBA", "Point Guard"]
print(df2)
print(df2.dtypes)

# print out all rows in table
print(df2.loc[2:])

# Print out a single row
print(df2.loc[1:1])

# print out only the column entries for the Names
print(df2.loc[0:, "Names"])
print(df2.loc[0:, ["Names", "Status"]])

# We can access only those values we are interested in
# For the following we will list the entries associated with the dead players
dfDead = df2[df2["Status"] == "Dead"]
print(dfDead)

dfNumbers = df2[df2["Status"] == "Dead"]["Numbers"]
print(dfNumbers)
