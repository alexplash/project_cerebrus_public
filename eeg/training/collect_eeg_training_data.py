
import time
import os
import numpy as np
import sys

EEG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, EEG_DIR)

from utils.eeg_streamer import EEGStreamer

TRAINING_DATA_LABELS = [
    "tilt_left", # turn left
    "tilt_right", # turn right
    "jaw_clench", # forward
    "raise_eyebrows", # stop
    "none" # nothing
]

def main():
    print("Enter the training data category being collected...")
    print(f"Options: {list(TRAINING_DATA_LABELS)}")
    user_input = input(">")
    
    user_input = user_input.strip()
    if user_input not in TRAINING_DATA_LABELS:
        raise ValueError("invalid trainig data label")
    
    record_time_s = int(input("Enter recording time in seconds: "))
    
    eeg_streamer = EEGStreamer()
    eeg_streamer.start()
    
    start = time.perf_counter()
    
    print("Recording Started...")
    
    all_data = []
    
    while True:
        data = eeg_streamer.pull_chunk()
        if data is not None:
            all_data.append(data)
        
        if time.perf_counter() - start >= record_time_s:
            break
        
        time.sleep(0.1)
    
    if not all_data:
        raise RuntimeError("No EEG Data was collected")
    
    full_data = np.vstack(all_data)
    
    os.makedirs("training_data", exist_ok = True)
    
    output_path = f"training_data/{user_input}.npy"
    np.save(output_path, full_data)
    
    print(f"Saved EEG data to: {output_path}")
    print(f"Label: {user_input}")
    print(f"Raw EEG shape: {full_data.shape}")

if __name__ == "__main__":
    main()
    