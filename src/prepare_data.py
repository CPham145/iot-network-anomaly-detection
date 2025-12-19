from pathlib import Path
import pandas as pd

IN_PATH = Path("data/iot_sample.csv")
OUT_PATH = Path("data/iot_sample_clean.csv")

df = pd.read_csv(IN_PATH)

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Optional: also strip label values if they are strings
if "Label" in df.columns and df["Label"].dtype == "object":
    df["Label"] = df["Label"].str.strip()

df.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH, "shape:", df.shape)
print("Label counts (top):")
print(df["Label"].value_counts().head())
