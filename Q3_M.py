import pandas as pd
import matplotlib.pyplot as plt

files = {
    "features_short.csv": "Short Window (0.5s / 125 samples)",
    "features_medium.csv": "Medium Window (1.0s / 250 samples)",
    "features_long.csv": "Long Window (2.5s / 625 samples)"
}

MAX_TIME_SEC = 45.0
ROLLING_STEPS = 20  # Number of feature updates to calculate variance across (~2 seconds)

# Create a figure with two vertically stacked subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for file, label in files.items():
    df = pd.read_csv(file)
        
    # Sort chronologically and normalize time
    df = df.sort_values(by="timestamp")
    df["time"] = df["timestamp"] - df["timestamp"].iloc[0]

    # Filter by time threshold
    df_filtered = df[df["time"] <= MAX_TIME_SEC].copy()
    
    # Compute rolling variance over the last N steps
    df_filtered["rolling_var"] = df_filtered["C3_log_mu"].rolling(window=ROLLING_STEPS).var()
    
    # Top Plot: Trajectories
    ax1.plot(df_filtered["time"], df_filtered["C3_log_mu"], label=label, alpha=0.85)
    
    # Bottom Plot: Rolling Variance
    ax2.plot(df_filtered["time"], df_filtered["rolling_var"], label=f"{label} Variance", alpha=0.85)

# Top Plot
ax1.set_ylabel("C3 Log10 Mu Power")
ax1.set_title("C3 Log Mu Power Trajectories (First 45 Seconds)")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.6)

# Bottom Plot
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Rolling Variance ($\sigma^2$)")
ax2.set_title(f"Rolling Feature Variance (Window = {ROLLING_STEPS} steps)")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()