import numpy as np
import pandas as pd

#create a panda series with index Titles
series1 = pd.Series([12,14,17,19,20], index = ["one", "two", "three", "four", "five"])
print(series1)


series2 = pd.Series([33, 22, 11, 10])
print (series2)

workout = {"Mon" : "Legs", "Tue" : "Core", "Wed": "Biceps", "Thur" : "Straight Fucking", "Fri" : "Leg"}
sWorkout = pd.Series(workout)
print(sWorkout)