import pandas as pd

dfBB = pd.read_csv("Data/mlb_salaries.csv")

print(dfBB.head().to_string())
print(dfBB.head())

dfTOR = dfBB[dfBB['teamid'] == 'TOR']

print(dfTOR.head().to_string())

dfAGG = dfTOR.aggregate("count")
print("\n Count is $%.2f" % dfAGG)

maxSal = dfTOR["salary"].aggregate("count")
print("\n Max salary is $%.2f" % maxSal)

maxPlay = dfTOR[dfTOR["salary"] == maxSal]
print("\nPlayer %s has max salary $%.2f" % (maxPlay["player_name"].values[0], maxSal))
