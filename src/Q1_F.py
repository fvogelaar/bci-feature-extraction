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
WINDOW_STEP = 25    # as specified by assignment 1
WINDOW_SIZE = 625   # (125, 250, or 625)

window = SlidingWindow(size=WINDOW_SIZE, step=WINDOW_STEP)
feature = BandPower(fs=FS, band=(8, 12))
logger = FeatureLogger("features.csv")

if USE_SIMULATION:
    print("using simulation data")
    signal = np.random.randn(FS * 5, N_CHANNELS)

    first_window_received = False
    # start the clock before window.update
    stream_start_time = time.monotonic()
    for sample in signal:
        win = window.update(sample)
        if win is not None:
            if not first_window_received:
                observed_delay = time.monotonic() - stream_start_time
                print(
                    f"First valid frame received after "
                    f"{observed_delay:.3f} seconds"
                )
                first_window_received = True
                # Note: observed delay for simulation = 0
                # because all samples arrive simultaneously
            feats = feature.compute(win)
            logger.log(feats)
            print("Features:", feats)
    print(f"Overlap percentage: {100*(1-(WINDOW_STEP/WINDOW_SIZE))}")

else:
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
    while True:
        # pull the samples & handle time out
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is None:
            print("No LSL sample received for 1 second.")
            continue
        eeg_data = sample[:8]
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
            print("Features:", feats)
        print(f"Overlap percentage: {100*(1-(WINDOW_STEP/WINDOW_SIZE))}")
