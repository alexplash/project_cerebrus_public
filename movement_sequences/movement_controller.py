
import threading
import time
import json
import os
from typing import Dict, Any
import socketio
from dotenv import load_dotenv

load_dotenv()

SEQ_FILE = "sequence.json"
WS_URL = os.getenv("WS_URL")

def load_pose_file(pose_file) -> Dict[str, Dict[str, int]]:
    if not os.path.exists(pose_file):
        return {}
    with open(pose_file, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def bus_set_pose(board, pose: Dict[int, int], duration_s: float) -> None:
    cmd = [[int(sid), int(pose[sid])] for sid in sorted(pose.keys(), key=int)]
    board.bus_servo_set_position(duration_s, cmd)


def run_sequence(board, sequence_name: str) -> None:
    dir = "movement_sequences"
    
    pose_file = f"{sequence_name}.json"
    pose_file = os.path.join(dir, pose_file)
    pose_data = load_pose_file(pose_file)

    seq_full = load_json(os.path.join(dir, SEQ_FILE))["sequences"]
    seq = [x for x in seq_full if x["name"] == sequence_name][0]
    steps = seq.get("steps", [])

    for i, step in enumerate(steps):
        pose_name = step.get("pose")
        duration = float(step.get("duration"))

        pose = pose_data.get(pose_name)
        if pose is None:
            raise ValueError(f"Pose '{pose_name}' not found in {pose_file}")

        print(f"[{sequence_name}] step {i+1}/{len(steps)} pose='{pose_name}' duration={duration:.2f}s")
        bus_set_pose(board, pose, duration)
        time.sleep(duration + 0.05)


def default(board):
    run_sequence(board, "default")


def forward(board):
    run_sequence(board, "forward")


def turn_left(board):
    run_sequence(board, "turn_left")


def turn_right(board):
    run_sequence(board, "turn_right")
    

class RobotController:
    def __init__(self, board):
        self.board = board
        self.latest_command = "default"
        self.active_command = None
        self.running = False
        
        self.command_lock = threading.Lock()

        self.listener_thread = threading.Thread(target=self._socket_loop, daemon=True)
        self.movement_thread = threading.Thread(target=self._move_loop, daemon=True)

        self.valid_commands = {"default", "forward", "turn_left", "turn_right"}
        
        self.eeg_robot_commands = {
            "left_clench": "turn_left",
            "right_clench": "turn_right",
            "bottom_clench": "default",
            "jaw_clench": "forward",
            "none": "continue"
        }
        
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1
        )
        
        self._register_socket_handlers()
    
    def _register_socket_handlers(self):
        @self.sio.event
        def connect():
            print("connected to WS server")
            self.sio.emit("register", "robot")
            print("registering as robot")
        
        @self.sio.event
        def disconnect():
            print("disconnected from WS server")
            self._set_latest_command("default")
        
        @self.sio.on("robot-connected")
        def on_robot_connected():
            print("successfully connected to WS server. robot connected")
        
        @self.sio.on("eeg-ready")
        def on_eeg_ready():
            print("eeg streamer is ready")
        
        @self.sio.on("eeg-disconnected")
        def on_eeg_disconnected():
            print("eeg streamer has disconnected")
        
        @self.sio.on("command")
        def on_command(data):
            print(f"received command from WS server: {data}")
            
            if isinstance(data, dict):
                received_command = data.get("command")
            else:
                received_command = data
            
            if received_command is None:
                return 
            
            if received_command in self.eeg_robot_commands:
                self._set_latest_command(self.eeg_robot_commands.get(received_command))
                
        
    def _set_latest_command(self, command):
        with self.command_lock:
            if command != self.latest_command:
                print(f"new command received: {command}")
                self.latest_command = command
    
    def _get_latest_command(self):
        with self.command_lock:
            return self.latest_command

    def start(self):
        self.running = True
        
        default(self.board)
        time.sleep(1)
        
        self.listener_thread.start()
        self.movement_thread.start()

    def stop(self):
        self.running = False
        
        try:
            if self.sio.connected:
                self.sio.disconnect()
        except Exception as e:
            print(f"error disconnecting from WS server: {e}")
    
    def _socket_loop(self):
        if not WS_URL:
            print("ERROR: WS_URL is not set")
            self._set_latest_command("default")
            self.stop()
        
        while self.running:
            try:
                print("connecting to WS server")
                self.sio.connect(WS_URL)
                self.sio.wait()
            except Exception as e:
                print(f"socket connection error: {e}")
                self._set_latest_command("default")
                self.stop()
            

    def _move_loop(self):
        while self.running:
            latest_command = self._get_latest_command()
            
            if self.active_command != latest_command and latest_command != "continue":
                print(f"switching command: {self.active_command} -> {latest_command}")
                self.active_command = latest_command
            
            if self.active_command == "default":
                default(self.board)
                time.sleep(0.05)
            
            elif self.active_command == "forward":
                forward(self.board)
                time.sleep(0.05)
            
            elif self.active_command == "turn_left":
                turn_left(self.board)
                time.sleep(0.05)
            
            elif self.active_command == "turn_right":
                turn_right(self.board)
                time.sleep(0.05)
            
            else:
                time.sleep(0.05)
            
            
                
    
    def manual_input(self, stance, iterations = 1):
        if stance == "default":
            default(self.board)
        elif stance == "forward":
            for _ in range(iterations):
                forward(self.board)
        elif stance == "turn_left":
            for _ in range(iterations):
                turn_left(self.board)
        elif stance == "turn_right":
            for _ in range(iterations):
                turn_right(self.board)
        else:
            print(f"ERROR: {stance} is not valid")
                    
                
        