import time
import signal
import ros_robot_controller_sdk as rrc

board = rrc.Board()
board.enable_reception(True)

start = True
def Stop(signum, frame):
    global start
    start = False
    print("\n🛑 STOPPING... (Ctrl+C received)\n", flush=True)

signal.signal(signal.SIGINT, Stop)

def clamp(x, lo=0, hi=1000):
    return max(lo, min(hi, x))

def read_pos(sid: int):
    p = board.bus_servo_read_position(sid)
    if not p:
        return None
    return p[0]

def countdown(seconds: int, label: str):
    for i in range(seconds, 0, -1):
        if not start:
            return
        print(f"  {label} in {i}...", flush=True)
        time.sleep(1)

if __name__ == "__main__":
    STEP = 150
    MOVE_T = 0.8
    PRE_WAIT = 2
    POST_WAIT = 2

    for sid in range(1, 25):
        if not start:
            break

        print("\n" + "=" * 60, flush=True)
        print(f"✅ NOW TESTING SERVO ID: {sid:02d}", flush=True)
        print("=" * 60, flush=True)

        base = read_pos(sid)
        if base is None:
            print(f"❌ ID {sid:02d}: NO RESPONSE (skipping)", flush=True)
            continue

        a = clamp(base + STEP)
        b = clamp(base - STEP)
        target = a if a != base else b

        print(f"📍 Base position:   {base}", flush=True)
        print(f"🎯 Target position: {target}   (STEP={STEP})", flush=True)
        print(f"⏱  Move duration:   {MOVE_T:.1f}s", flush=True)

        countdown(PRE_WAIT, "MOVE")
        if not start:
            break

        print("🚀 MOVING TO TARGET...", flush=True)
        board.bus_servo_set_position(MOVE_T, [[sid, target]])
        time.sleep(MOVE_T + 0.3)

        print("↩️  RETURNING TO BASE...", flush=True)
        board.bus_servo_set_position(MOVE_T, [[sid, base]])
        time.sleep(MOVE_T + 0.3)

        print("✅ DONE WITH THIS SERVO.", flush=True)
        countdown(POST_WAIT, "NEXT SERVO")
        if not start:
            break

    print("\n🧯 Stopping all servos 1–24...", flush=True)
    board.bus_servo_stop(list(range(1, 25)))
    print("🏁 Done.", flush=True)