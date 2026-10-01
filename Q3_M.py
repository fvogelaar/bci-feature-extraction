import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS = 250  
C3_COL = "C3_log_mu"

files = {
    "features_short.csv": "Short Window (125 samples / 0.5s)",
    "features_medium.csv": "Medium Window (250 samples / 1.0s)",
    "features_long.csv": "Long Window (625 samples / 2.5s)"
}

plt.figure(figsize=(12, 6))
MAX_TIME_SEC = 45.0
for file, label in files.items():
    df = pd.read_csv(file)
    
    df['time'] = df["timestamp"] - df["timestamp"].iloc[0]
    print(df["C3_log_mu"])
    # Drop NaNs and filter to 45 seconds
    df_clean = df.dropna(subset=["C3_log_mu"])
    df_filtered = df_clean[df_clean["time"] <= MAX_TIME_SEC]
    
    plt.plot(df_filtered["time"], df_filtered["C3_log_mu"], label=label, alpha=0.8)

plt.xlabel("Time (s)")
plt.ylabel("C3 Log10 Mu Power")
plt.title("C3 Log Mu Power Trajectories (First 45 Seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()