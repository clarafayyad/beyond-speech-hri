import pandas as pd

# Load your CSV
df = pd.read_csv("pilot.csv")

if "clue_id" in df.columns:
    print("clue_id column already exists. Please remove it before running this script.")
    exit(1)

# Create clue_id based on row number (starting at 1)
df["clue_id"] = range(1, len(df) + 1)

# Save back to CSV
df.to_csv("pilot.csv", index=False)