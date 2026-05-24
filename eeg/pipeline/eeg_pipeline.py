
import os
import sys
import time

import numpy as np
import socketio
import torch
from dotenv import load_dotenv

EEG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, EEG_DIR)

from utils.eeg_mne_utils import preprocess_eeg_array_with_mne
from utils.eeg_streamer import EEGStreamer
from training.models import EEGSubtractiveConv2D

load_dotenv()

WS_URL = os.getenv("WS_URL")
MODEL_WEIGHTS_PATH = "../training/model_weights/EEGSubtractiveConv2D.pt"
NUM_RAW_CHANNELS = 4

SAMPLES_PER_CLASSIFICATION = 100

LABELS_MAP = {
    1: "left_clench",
    2: "right_clench",
    3: "bottom_clench",
    4: "jaw_clench",
    0: "none"
}

sio = socketio.Client(
    reconnection=True,
    reconnection_attempts=5,
    reconnection_delay=1
)

@sio.event
def connect():
    print("connected to server")
    sio.emit("register", "eeg")
    print("registered as eeg")

@sio.event
def disconnect():
    print("disconnected from server")

def init_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EEGSubtractiveConv2D()
    
    state_dict = torch.load(
        MODEL_WEIGHTS_PATH,
        map_location = device
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded model weights from {MODEL_WEIGHTS_PATH}")
    print(f"Using device: {device}")
    
    return model, device

def main():
    
    print(f"connecting to {WS_URL}")
    sio.connect(WS_URL)
    
    model, device = init_model()
    
    eeg_streamer = EEGStreamer()
    eeg_streamer.start()
    
    print("Recording Started...")
    
    curr_samples = np.empty((0, NUM_RAW_CHANNELS))
    
    while True:
        data = eeg_streamer.pull_chunk()
        if data is not None:
            curr_samples = np.vstack([curr_samples, data])
            
        if (curr_samples.shape[0] >= SAMPLES_PER_CLASSIFICATION):
            window = curr_samples[-SAMPLES_PER_CLASSIFICATION:]
            
            clean_eeg = preprocess_eeg_array_with_mne(
                data = window,
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
            
            x = torch.tensor(clean_eeg, dtype = torch.float32)
            x = x.unsqueeze(0)
            x = x.to(device)
            
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(logits, dim = 1)
                pred_value = int(pred.squeeze().item())
            
            command = LABELS_MAP.get(pred_value)
            sio.emit("command", command)
            
            curr_samples = np.empty((0, NUM_RAW_CHANNELS))
        
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ending stream")
    