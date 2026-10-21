
import pandas as pd
from pathlib import Path

# Project folders
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "household_power_consumption.txt"
OUTPUT = BASE / "data" / "hourly_consumption.csv"

# Read the original dataset
df = pd.read_csv(
    DATA,
    sep=";",
    na_values=["?"],
    low_memory=False
)

# Convert power readings to numbers
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], errors="coerce"
)

# Combine date and time
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

# Keep valid timestamps and power readings
df = df.dropna(subset=["datetime", "Global_active_power"])
df = df.sort_values("datetime").set_index("datetime")

# Average power (kW) for each hour
hourly = df["Global_active_power"].resample("h").mean()

# Convert average kW to energy used per hour (kWh)
hourly = hourly.rename("consumption_kwh")

# Fill missing hourly readings using time interpolation
hourly = hourly.ffill(limit=2)

# Create time-based features
result = hourly.to_frame()
result["hour"] = result.index.hour
result["day_of_week"] = result.index.dayofweek

# Target: consumption in the NEXT hour
result["target_kwh"] = result["consumption_kwh"].shift(-1)

# Remove rows without a target or usable readings
result = result.dropna()

# Save prepared data
result.to_csv(OUTPUT)

print("Data preparation complete!")
print("Rows and columns:", result.shape)
print(result.head())
print("Saved to:", OUTPUT)
