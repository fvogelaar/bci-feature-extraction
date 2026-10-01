"""
Real-time feature extraction pipeline
"""

from pylsl import StreamInlet, resolve_byprop
from src.window import SlidingWindow
from src.bandpower import BandPower
from src.log import FeatureLogger
import numpy as np
import time

FS = 250
N_CHANNELS = 8
USE_SIMULATION = True
WINDOW_STEP = 25    
WINDOW_SIZE = 625   # (125, 250, or 625)

# not really sure about whether this indexing is correct
C3_IDX, C4_IDX = 1, 3

window = SlidingWindow(size=WINDOW_SIZE, step=WINDOW_STEP)
# mu band is 8-12
feature = BandPower(fs=FS, band=(8, 12))
logger = FeatureLogger("features_long.csv")

streams = resolve_byprop(
    "type",
    "EEG",
    timeout=5
    )

if not streams:
    raise RuntimeError(
        "No EEG type LSL stream found. "
    )

print(f"Stream name: {streams[0].name()}")
print(f"# channels published: {streams[0].channel_count()}")

inlet = StreamInlet(streams[0])
first_window_received = False
while True:
    # pull the samples & handle time out
    sample, timestamp = inlet.pull_sample(timeout=1.0)
    if sample is None:
        print("No LSL sample received for 1 second.")
        continue

    eeg_data = sample[:N_CHANNELS]
    eeg_np = np.array(eeg_data, dtype=float)
    # start the clock before window.update
    stream_start_time = time.monotonic()
    win = window.update(eeg_np)


    if win is not None:
        if not first_window_received:
            observed_delay = time.monotonic() - stream_start_time
            print(
                f"First valid frame received after "
                f"{observed_delay:.3f} seconds"
                )
            first_window_received = True
        feats = feature.compute(win)

        logger.log(feats)
        raw_band_power = feature.compute(win)
        # Extract C3 and C4 channels
        motor_power = raw_band_power[[C3_IDX, C4_IDX]]

        # log transformation
        log_features = np.log10(motor_power + 1e-10)
        logger.log(log_features, timestamp=timestamp)
        print(f"Logged at t={timestamp:.4f} | Features (C3, C4 log-mu power): {log_features}")