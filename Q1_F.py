"""
Real-time feature extraction pipeline
"""

from pylsl import StreamInlet, resolve_byprop, resolve_streams
from src.window import SlidingWindow
from src.bandpower import BandPower
from src.log import FeatureLogger
import numpy as np
import time

FS = 250
N_CHANNELS = 8
USE_SIMULATION = True
WINDOW_STEP = 25    # as specified by assignment 1
WINDOW_SIZE = 250   # (125, 250, or 625)

window = SlidingWindow(size=WINDOW_SIZE, step=WINDOW_STEP)
feature = BandPower(fs=FS, band=(8, 12))
logger = FeatureLogger("features.csv")

if USE_SIMULATION:
    print("using simulation data")
    stream_start_time = time.monotonic()
    signal = np.random.randn(FS * 5, N_CHANNELS)

    first_window_received = False

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
                # Note: observed delay for np-array simulation = 0
                # because all samples arrive simultaneously
            feats = feature.compute(win)
            logger.log(feats)
            print("Features:", feats)
    print(f"Overlap percentage: {100*(1-(WINDOW_STEP/WINDOW_SIZE))}")

else:
    streams = resolve_streams()
    if not streams:
        raise RuntimeError(
            "No LSL stream found. "
        )

    print(f"Stream name: {streams[0].name()}")
    print(f"# channels published: {streams[0].channel_count()}")

    first_window_received = False

    inlet = StreamInlet(streams[0])

    stream_start_time = time.monotonic()

    while True:
        # pull the samples & handle time out
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is None:
            print("No LSL sample received for 1 second.")
            continue
        eeg_data = sample[:8]
        eeg_np = np.array(eeg_data, dtype=float)
        
        win = window.update(eeg_np)
        first_window_received = False
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
            #print("Features:", feats)
