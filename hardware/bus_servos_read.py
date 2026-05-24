import time
import signal
import ros_robot_controller_sdk as rrc

board = rrc.Board()
board.enable_reception(True)

start = True
def Stop(signum, frame):
    global start
    start = False
    print("Stopping...")

signal.signal(signal.SIGINT, Stop)

def read_one(board, sid: int):
    pos = board.bus_servo_read_position(sid)
    if not pos:
        print(f"[..] id={sid:02d} no response")
        return False

    pos_v = pos[0] if isinstance(pos, list) else pos
    print(f"id={sid:02d}  pos={pos_v}")
    return True

if __name__ == "__main__":
    try:
        found = []
        for sid in range(1, 25):
            if not start:
                break
            if read_one(board, sid):
                found.append(sid)

        print("FOUND:", found)
        if len(found) == 24:
            print("all servos found!")
    except KeyboardInterrupt:
        print("forced exit")