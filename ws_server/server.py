
import os
import uvicorn
import socketio

sio = socketio.AsyncServer(
    async_mode = "asgi",
    cors_allowed_origins = "*"
)

app = socketio.ASGIApp(sio)

eeg_socket = None
robot_socket = None

@sio.event
async def connect(sid, environ):
    print(f"client connected: {sid}")

@sio.event
async def disconnect(sid):
    global eeg_socket, robot_socket
    
    print(f"client disconnected: {sid}")
    
    if sid == robot_socket:
        robot_socket = None
        print("robot disconnected")
        
        if eeg_socket:
            await sio.emit("robot-disconnected", to=eeg_socket)
    
    if sid == eeg_socket:
        eeg_socket = None
        print("eeg disconnected")
        
        if robot_socket:
            await sio.emit("eeg-disconnected", to=robot_socket)

@sio.on("register")
async def register(sid, role):
    global eeg_socket, robot_socket
    
    if role == "eeg":
        eeg_socket = sid
        print("eeg streamer connected")
        await sio.emit("eeg-connected", to=sid)
    
    elif role == 'robot':
        robot_socket = sid
        print("robot connected")
        await sio.emit("robot-connected", to=sid)
    
    if eeg_socket and robot_socket:
        await sio.emit("robot-ready", to=eeg_socket)
        await sio.emit("eeg-ready", to=robot_socket)

@sio.on("command")
async def command(sid, command):
    
    if sid != eeg_socket:
        print("rejected command from non eeg socket")
        return
    
    if not robot_socket:
        print("no robot connected; command dropped")
        return
    
    print(f"forwarding eeg command to robot: {command}")
    await sio.emit("command", command, to=robot_socket)


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 3001))
    print(f"Signaling on :{PORT}")
    uvicorn.run(app, host = "0.0.0.0", port = PORT)
    
    