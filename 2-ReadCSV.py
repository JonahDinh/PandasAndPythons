import pandas as pd

myPlanes = pd.read_csv("Data/wwIIAircraft.csv")

print(myPlanes.head(5))
print(myPlanes.tail(5))

# Print out all the planes who's Country of Origin is the US
usPlanes = myPlanes[myPlanes['Country of Origin'] == 'US']
print(usPlanes)