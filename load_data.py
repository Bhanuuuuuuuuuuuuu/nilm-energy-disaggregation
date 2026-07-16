"""
Day 1 - Step 1 (fixed): Load, align, and explore UK-DALE house_1 data

Problem we're solving here:
Each channel was recorded on its own independent timer (~every 6 seconds),
so raw timestamps don't line up across channels. If we naively combine them,
we get a huge sparse table full of NaNs.

Fix: resample every channel onto the SAME regular time grid (every 6 seconds),
using the most recent known reading for each slot (forward-fill), so all
columns are aligned and comparable at every timestamp.
"""

import pandas as pd
import matplotlib.pyplot as plt

CHANNELS = {
    1: "aggregate",
    12: "fridge",
    13: "microwave",
    5: "washing_machine",
    10: "kettle",
}

RESAMPLE_INTERVAL = "6s"          # UK-DALE's native rate
MAX_GAP_FILL = "30s"              # don't forward-fill across gaps longer than this


def load_channel(channel_num, name):
    filepath = f"channel_{channel_num}.dat"
    df = pd.read_csv(
        filepath,
        sep=" ",
        header=None,
        names=["timestamp", name],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")
    # Some channels have duplicate timestamps - keep the last reading
    df = df[~df.index.duplicated(keep="last")]
    return df


print("Loading channels...")
data = {}
for ch_num, name in CHANNELS.items():
    print(f"  loading channel {ch_num} ({name})...")
    data[name] = load_channel(ch_num, name)
    print(f"    -> {len(data[name])} rows, "
          f"from {data[name].index.min()} to {data[name].index.max()}")

# ---- Resample each channel onto the same regular grid ----
print(f"\nResampling each channel to a common {RESAMPLE_INTERVAL} grid...")
resampled = {}
for name, df in data.items():
    # mean() collapses any readings that fall in the same 6-second bucket
    resampled[name] = df.resample(RESAMPLE_INTERVAL).mean()

# ---- Find the common overlapping period across all channels ----
common_start = max(df.index.min() for df in resampled.values())
common_end = min(df.index.max() for df in resampled.values())
print(f"\nCommon overlapping period across all channels: {common_start} to {common_end}")

# ---- Merge onto one shared timeline ----
merged = pd.concat(resampled.values(), axis=1)
merged = merged.sort_index()

# ---- Forward-fill small gaps only (don't invent long stretches of fake data) ----
max_fill_periods = int(pd.Timedelta(MAX_GAP_FILL) / pd.Timedelta(RESAMPLE_INTERVAL))
merged = merged.ffill(limit=max_fill_periods)

# ---- Drop rows where we still don't have the aggregate reading (our main input) ----
merged = merged.dropna(subset=["aggregate"])

print("\nFirst few rows of merged, aligned data:")
print(merged.head(10))
print("\nShape:", merged.shape)
print("\nMissing values per column (after alignment + fill):")
print(merged.isna().sum())
print("\n% missing per column:")
print((merged.isna().mean() * 100).round(2))

# ---- Quick plot: first 24 hours ----
window = merged.loc[common_start: common_start + pd.Timedelta(hours=24)]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(window.index, window["aggregate"], color="black")
axes[0].set_title("Whole-house aggregate power (W)")

axes[1].plot(window.index, window["fridge"], color="blue")
axes[1].set_title("Fridge power (W)")

axes[2].plot(window.index, window["kettle"], color="red")
axes[2].set_title("Kettle power (W)")

plt.tight_layout()
plt.savefig("day1_exploration.png", dpi=120)
print("\nSaved plot to day1_exploration.png -- open it and take a look!")

# ---- Save cleaned data for tomorrow ----
merged.to_csv("merged_house1.csv")
print("Saved merged, aligned data to merged_house1.csv")