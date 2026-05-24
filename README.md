# Project Cerebrus

Project Cerebrus is an EEG-controlled robotics project that connects a PLUX NeuroBIT EEG hardware kit to a HiWonder AiNex robot. The system streams live EEG from a MacBook, classifies the signal into movement commands, sends those commands through a websocket relay server, and executes movement sequences on the Raspberry Pi inside the robot.

The repository contains three main parts:

- AiNex robot control code, including hardware access and servo movement sequences.
- EEG streaming, preprocessing, training, and inference code for the PLUX NeuroBIT EEG setup.
- A websocket relay server that passes commands between the MacBook and the robot computer.

## System Overview

In production, the pipeline runs across three machines/processes:

1. The MacBook runs `eeg/pipeline/eeg_pipeline.py`.
   It connects to the PLUX/OpenSignals LSL stream, preprocesses EEG windows, runs a trained PyTorch classifier, and sends predicted commands to the websocket server.

2. The websocket server in `ws_server/` acts as a relay.
   It registers one EEG client and one robot client, then forwards classified EEG commands from the MacBook to the robot.

3. The Raspberry Pi on the AiNex robot runs `main.py`.
   It connects to the websocket server, receives commands, and drives the robot through predefined servo movement sequences.

## EEG Setup

The EEG pipeline was built for the PLUX NeuroBIT EEG hardware kit using OpenSignals with LSL streaming enabled.

The production EEG setup uses:

- Two EEG electrode channels placed on the frontal lobes.
- One reference electrode placed behind the left ear.
- Subtraction/reference preprocessing before classification.

The EEG code includes:

- `eeg/utils/eeg_streamer.py`  
  LSL stream discovery and live sample collection for the OpenSignals stream.

- `eeg/utils/eeg_mne_utils.py`  
  MNE-based EEG preprocessing utilities.

- `eeg/training/collect_eeg_training_data.py`  
  Data collection code for recording labeled EEG samples.

- `eeg/training/train_models.py`  
  Training code for the EEG classifiers.

- `eeg/training/models.py`  
  PyTorch model architectures tested for EEG classification.

- `eeg/pipeline/eeg_pipeline.py`  
  The production MacBook pipeline for live EEG streaming, classification, and websocket command emission.

## Robot Control

The robot side is built around the AiNex Raspberry Pi and its servo controller.

- `main.py`  
  Entry point for the Raspberry Pi on the robot.

- `hardware/ros_robot_controller_sdk.py`  
  Low-level serial SDK for communicating with the robot controller board.

- `hardware/bus_servos_read.py` and `hardware/bus_servo_turn.py`  
  Utility scripts for reading and testing bus servos.

- `movement_sequences/`  
  JSON pose and sequence files for robot movement.

- `movement_sequences/movement_controller.py`  
  Robot command loop. It receives commands from the websocket relay and maps EEG classifications to movement sequences.

Current command mapping:

| EEG classification | Robot command |
| --- | --- |
| `left_clench` | `turn_left` |
| `right_clench` | `turn_right` |
| `jaw_clench` | `forward` |
| `bottom_clench` | `default` |
| `none` | continue current movement |

## Websocket Server

The `ws_server/` folder contains the relay communication server between the MacBook and the robot computer.

- `ws_server/server.py`  
  Socket.IO ASGI server that registers EEG and robot clients, tracks connection state, and forwards commands from EEG to robot.

- `ws_server/requirements.txt`  
  Server dependencies.

- `ws_server/Procfile`  
  Deployment process definition.

## Installation

Create a Python virtual environment, then install the dependencies for the component you are running.

For the MacBook EEG pipeline:

```bash
pip install -r requirements.txt
```

For the robot Raspberry Pi:

```bash
pip install -r requirements_robot.txt
```

For the websocket server:

```bash
cd ws_server
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the websocket server URL:

```bash
WS_URL=http://localhost:3001
```

Use the deployed websocket server URL when running across machines.

## Running

Start the websocket relay:

```bash
cd ws_server
python server.py
```

Run the EEG pipeline on the MacBook:

```bash
python eeg/pipeline/eeg_pipeline.py
```

Run the robot controller on the Raspberry Pi:

```bash
python main.py
```

## Data and Model Files

Training data and model weights are intentionally excluded from version control:

- `eeg/training/training_data/`
- `eeg/training/model_weights/`

These files can include biometric EEG samples and derived trained models, so they should be treated as private unless intentionally released.

