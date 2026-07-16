import pandas as pd

df = pd.read_csv("channel_1.dat", sep=" ", header=None, names=["timestamp", "aggregate"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
print(df["timestamp"].min(), "to", df["timestamp"].max())