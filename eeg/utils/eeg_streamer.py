
import numpy as np
from pylsl import resolve_byprop, StreamInlet

CHANNELS = [0, 1, 2, 3]

class EEGStreamer:
    
    def __init__(self):
        self.channels = CHANNELS
        self.stream_name = "OpenSignals"
        self.stream = self._find_lsl_stream()
        self.inlet = None

    def _find_lsl_stream(self):
        print("searching for LSL stream...")
        
        streams = resolve_byprop("name", self.stream_name)
        
        if not streams:
            raise RuntimeError(
                "No LSL streams found. Make sure OpenSignals is running, "
                "LSL streaming is enabled, and acquisition has started."
            )
        
        print("\nFound Streams:")
        for i, stream in enumerate(streams):
            print(f"[{i}] name={stream.name()} type={stream.type()} channels={stream.channel_count()}")
        
        for stream in streams:
            if stream.channel_count() >= 4 and stream.name() == "OpenSignals":
                print(f"Using stream: {stream.name()}")
                return stream
        
        raise RuntimeError("Found LSL stream, but none had at least 4 channels.")
    
    def start(self):
        self.inlet = StreamInlet(self.stream)
        print(f"\nStreaming channels: {self.channels}")

    def pull_chunk(self):
        if self.inlet is None:
            raise RuntimeError("Must call start() before pull_chunk()")
        
        samples, _ = self.inlet.pull_chunk(timeout=1.0, max_samples=100)
        
        if not samples:
            print("No samples received...")
            return None
        
        samples = np.array(samples, dtype=np.float32)
        
        data = samples[:, self.channels]
        return data
    