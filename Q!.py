"""
Real-time feature extraction pipeline
"""

from pylsl import StreamInlet, resolve_stream
from src.window import SlidingWindow
from src.bandpower import BandPower
from src.log import FeatureLogger

FS = 250
N_CHANNELS = 8

streams = resolve_stream("type", "EEG")
inlet = StreamInlet(streams[0])

window = SlidingWindow(size=FS, step=FS // 4)
feature = BandPower(fs=FS, band=(8, 12))
logger = FeatureLogger("features.csv")

while True:
    sample, _ = inlet.pull_sample()
    win = window.update(sample)

    if win is not None:
        feats = feature.compute(win)
        logger.log(feats)
        print("Features:", feats)
