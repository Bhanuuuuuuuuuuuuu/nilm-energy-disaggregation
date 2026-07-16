import pandas as pd

df = pd.read_csv("featured_house1.csv", parse_dates=["timestamp"], index_col="timestamp")

print(df.index.min(), "to", df.index.max())

cutoff_date = "2016-08-01"

train = df.loc[:cutoff_date]
test = df.loc[cutoff_date:]

print("Train rows:", len(train), "| Test rows:", len(test))

train.to_csv("train_house1.csv")
test.to_csv("test_house1.csv")