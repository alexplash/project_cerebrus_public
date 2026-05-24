import time
import signal
import hardware.ros_robot_controller_sdk as rrc
from movement_sequences.movement_controller import RobotController

running = True
def handle_sigint(signum, frame):
    global running
    running = False
    print("\nStopping...", flush=True)

signal.signal(signal.SIGINT, handle_sigint)

def main():
    global running
    
    board = rrc.Board()
    board.enable_reception(True)
    time.sleep(0.2)

    robot_controller = RobotController(board)
    
    try:
        robot_controller.start()
        
        while running:
            time.sleep(0.1)
    finally:
        robot_controller.stop()
        time.sleep(0.2)

if __name__ == "__main__":
    main()