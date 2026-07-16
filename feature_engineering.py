import pandas as pd

df = pd.read_csv("merged_house1.csv")
print(df.dtypes)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# ---- Feature engineering ----
df["agg_roll_mean_5min"] = df["aggregate"].rolling("5min").mean()
df["agg_roll_mean_1min"] = df["aggregate"].rolling("1min").mean()
df["agg_roll_mean_30s"] = df["aggregate"].rolling("30s").mean()
df["agg_diff"] = df["aggregate"].diff()
df["agg_roll_std_5min"] = df["aggregate"].rolling("5min").std()

print(df.head())

# ---- Labels ----
df["kettle_on"] = (df["kettle"] > 500).astype(int)
print(df["kettle_on"].value_counts())

df["fridge_on"] = (df["fridge"] > 10).astype(int)
print(df["fridge_on"].value_counts())

# ---- Save ----
df.to_csv("featured_house1.csv")
print("Saved feature-engineered data to featured_house1.csv")