
import time
import os
import numpy as np
from ..utils.eeg_mne_utils import preprocess_eeg_array_with_mne
from ..utils.eeg_streamer import EEGStreamer

TRAINING_DATA_LABELS = [
    "left_clench", # turn left
    "right_clench", # turn right
    "bottom_clench", # forward
    "jaw_clench", # stop
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
    
    clean_eeg = preprocess_eeg_array_with_mne(
        data = full_data,
        sfreq = 100.0,
        electrode_cols = [1, 2],
        reference_col = 3,
        ch_names = ["electrode_1", "electrode_2"],
        input_units = "uV",
        output_units = "uV",
        l_freq = 1.0,
        h_freq = 35.0,
        notch_freq = None
    )
    
    os.makedirs("training_data", exist_ok = True)
    
    output_path = f"training_data/{user_input}.npy"
    np.save(output_path, clean_eeg)
    
    print(f"Saved clean EEG data to: {output_path}")
    print(f"Label: {user_input}")
    print(f"Raw EEG shape: {full_data.shape}")
    print(f"Clean EEG shape: {clean_eeg.shape}")

if __name__ == "__main__":
    main()
    