
import torch
from models import EEGFlattenedMLP, EEGTransformerEncoder, EEGConv2D, EEGSubtractiveConv2D
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys

EEG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, EEG_DIR)

from utils.eeg_mne_utils import preprocess_eeg_array

LABELS_MAP = {
    "tilt_left": 1,
    "tilt_right": 2,
    "jaw_clench": 3,
    "raise_eyebrows": 4,
    "none": 0
}

class EEGDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]['data']
        label = self.data[idx]['label']
        
        sample = torch.tensor(sample, dtype = torch.float32)
        label = torch.tensor(label, dtype = torch.long)
        
        return sample, label

def gen_train_val_datasets(trial_idx):
    data_dir = "training_data"
    window_size = 1000
    stride = 250
    train_data_list = []
    val_data_list = []
    
    for class_name, label in LABELS_MAP.items():
        file_path = os.path.join(data_dir, f"{class_name}.npy")
        
        array = np.load(file_path)  # [timesteps, 2]
        num_timesteps = array.shape[0]
            
        if trial_idx == 0:
            # val = last 10%, train = first 90%
            val_start = int(num_timesteps * 0.9)
            val_end = num_timesteps
            
            train_arrays = [array[:val_start]]
            val_arrays = [array[val_start:val_end]]
        
        elif trial_idx == 1:
            # val = first 5% + last 5%, train = middle 90%
            left_val_end = int(num_timesteps * 0.05)
            right_val_start = int(num_timesteps * 0.95)
            
            train_arrays = [array[left_val_end:right_val_start]]
            val_arrays = [array[:left_val_end], array[right_val_start:]]
        
        elif trial_idx == 2:
            # val = 30-40%, train = before & after
            val_start = int(num_timesteps * 0.3)
            val_end = int(num_timesteps * 0.4)
            
            train_arrays = [array[:val_start], array[val_end:]]
            val_arrays = [array[val_start:val_end]]
        
        elif trial_idx == 3:
            # val = 60-70%, train = before & after
            val_start = int(num_timesteps * 0.6)
            val_end = int(num_timesteps * 0.7)
            
            train_arrays = [array[:val_start], array[val_end:]]
            val_arrays = [array[val_start:val_end]]
        
        for train_array in train_arrays:
            for start in range(0, train_array.shape[0] - window_size + 1, stride):
                end = start + window_size
                window = train_array[start:end]
                
                clean_eeg = preprocess_eeg_array(
                    data = window,
                    electrode_cols = [1, 2],
                    reference_col = 3,
                )
                
                train_data_list.append({"data": clean_eeg, "label": label})
        
        for val_array in val_arrays:
            for start in range(0, val_array.shape[0] - window_size + 1, stride):
                end = start + window_size
                window = val_array[start:end]
                
                clean_eeg = preprocess_eeg_array(
                    data = window,
                    electrode_cols = [1, 2],
                    reference_col = 3,
                )
                
                val_data_list.append({"data": clean_eeg, "label": label})
    
    return EEGDataset(train_data_list), EEGDataset(val_data_list)

def train(model_class, train_dataset, trial_idx):
    
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class().to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    num_epochs = 20
    
    print(f"starting training for {model_class}. trial idx {trial_idx}")
    print("------------------------------------------------------")
    
    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}/{num_epochs}")
        
        model.train()
        
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for x, y in train_loader:
            x = x.to(device) # [B, 1000, 2]
            y =  y.to(device) # [B]
            
            logits = model(x)
            loss = loss_fn(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            
            preds = torch.argmax(logits, dim = 1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += y.size(0)
        
        avg_epoch_loss = epoch_loss / epoch_total
        epoch_acc = epoch_correct / epoch_total
    
        print(f"epoch loss = {avg_epoch_loss}")
        print(f"epoch accuracy = {epoch_acc}")
    
    print(f"training completed for {model_class}. trial idx {trial_idx}")
    print("------------------------------------------------------")
    print("\n\n")
    
    return model      

def validate(model_class, model, val_dataset, trial_idx):
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"starting validation for {model_class}. trial idx {trial_idx}")
    print("------------------------------------------------------")
    
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device) # [B, 1000, 2]
            y =  y.to(device) # [B]
            
            logits = model(x)
            
            preds = torch.argmax(logits, dim = 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct / total

    print(f"model accuracy = {acc}")
    
    print(f"validation completed for {model_class}. trial idx {trial_idx}")
    print("------------------------------------------------------")
    print("\n\n")
    
    return acc
        

def main():
    
    trial_indices = range(4)
    
    model_classes = [EEGFlattenedMLP, EEGTransformerEncoder, EEGConv2D, EEGSubtractiveConv2D]
    
    os.makedirs("model_weights", exist_ok=True)
    
    model_accuracy_map = {
        "model_name": [],
        "avg_accuracy": [],
        "best_accuracy": [],
        "best_trial_idx": []
    }
    
    for model_class in model_classes:
        model_name = model_class.__name__
        
        accuracies = []
        best_accuracy = -1.0
        best_trial_idx = None
        best_state_dict = None
        
        for trial_idx in trial_indices:
            train_dataset, val_dataset = gen_train_val_datasets(trial_idx)
            
            model = train(model_class, train_dataset, trial_idx)
            accuracy_score = validate(model_class, model, val_dataset, trial_idx)
            
            accuracies.append(accuracy_score)
            
            if accuracy_score > best_accuracy:
                best_accuracy = accuracy_score
                best_trial_idx = trial_idx
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        save_path = os.path.join("model_weights", f"{model_name}.pt")
        torch.save(best_state_dict, save_path)
        
        model_accuracy_map["model_name"].append(model_name)
        model_accuracy_map["avg_accuracy"].append(avg_accuracy)
        model_accuracy_map["best_accuracy"].append(best_accuracy)
        model_accuracy_map["best_trial_idx"].append(best_trial_idx)
        
        print(f"saved best weights for {model_name} to {save_path}")
        print(f"avg accuracy = {avg_accuracy}")
        print(f"best accuracy = {best_accuracy}")
        print(f"best trial idx = {best_trial_idx}")
    
    df = pd.DataFrame(model_accuracy_map)
    df.to_csv("model_weights/model_accuracy.csv", index=False)
    


if __name__ == "__main__":
    main()