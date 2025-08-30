from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

rooms = {}
users = {}

@app.route('/health')
def health():
    return {'status': 'healthy'}

@socketio.on('connect')
def handle_connect():
    user_id = str(uuid.uuid4())
    users[request.sid] = {
        'id': user_id,
        'role': None,
        'room': None
    }
    emit('connected', {'userId': user_id})
    print(f'User {user_id} connected')

@socketio.on('disconnect')
def handle_disconnect():
    user = users.get(request.sid)
    if user and user['room']:
        room_id = user['room']
        leave_room(room_id)
        
        if room_id in rooms:
            if user['role'] == 'host':
                socketio.emit('host_disconnected', room=room_id)
                del rooms[room_id]
            else:
                rooms[room_id]['viewers'] = [v for v in rooms[room_id]['viewers'] if v != request.sid]
                socketio.emit('viewer_left', {'userId': user['id']}, room=room_id)
    
    if request.sid in users:
        del users[request.sid]
    print(f'User disconnected')

@socketio.on('create_room')
def handle_create_room():
    room_id = str(uuid.uuid4())[:8]
    join_room(room_id)
    
    rooms[room_id] = {
        'host': request.sid,
        'viewers': []
    }
    
    users[request.sid]['role'] = 'host'
    users[request.sid]['room'] = room_id
    
    emit('room_created', {'roomId': room_id})
    print(f'Room {room_id} created')

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data['roomId']
    
    if room_id not in rooms:
        emit('error', {'message': 'Room not found'})
        return
    
    join_room(room_id)
    rooms[room_id]['viewers'].append(request.sid)
    users[request.sid]['role'] = 'viewer'
    users[request.sid]['room'] = room_id
    
    emit('joined_room', {'roomId': room_id})
    socketio.emit('viewer_joined', {'userId': users[request.sid]['id']}, room=room_id)
    print(f'User joined room {room_id}')

@socketio.on('offer')
def handle_offer(data):
    room_id = users[request.sid]['room']
    if room_id and users[request.sid]['role'] == 'host':
        # Send offer to specific viewer if specified, otherwise to the most recent viewer
        target_viewer = data.get('to')
        if target_viewer:
            # Send to specific viewer
            for viewer_sid in rooms[room_id]['viewers']:
                if users[viewer_sid]['id'] == target_viewer:
                    socketio.emit('offer', {
                        'offer': data['offer'],
                        'from': users[request.sid]['id']
                    }, room=viewer_sid)
                    break
        else:
            # Send to most recent viewer (last one in the list)
            if rooms[room_id]['viewers']:
                latest_viewer_sid = rooms[room_id]['viewers'][-1]
                socketio.emit('offer', {
                    'offer': data['offer'],
                    'from': users[request.sid]['id']
                }, room=latest_viewer_sid)

@socketio.on('answer')
def handle_answer(data):
    room_id = users[request.sid]['room']
    if room_id and users[request.sid]['role'] == 'viewer':
        # Send answer back to host
        host_sid = rooms[room_id]['host']
        socketio.emit('answer', {
            'answer': data['answer'],
            'from': users[request.sid]['id']
        }, room=host_sid)

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    room_id = users[request.sid]['room']
    if room_id:
        # Send ICE candidate to the appropriate peer
        if users[request.sid]['role'] == 'host':
            # Send to specific viewer if specified
            target_viewer = data.get('to')
            if target_viewer:
                # Send to specific viewer
                for viewer_sid in rooms[room_id]['viewers']:
                    if users[viewer_sid]['id'] == target_viewer:
                        socketio.emit('ice_candidate', {
                            'candidate': data['candidate'],
                            'from': users[request.sid]['id']
                        }, room=viewer_sid)
                        break
            else:
                # Send to all viewers (fallback)
                for viewer_sid in rooms[room_id]['viewers']:
                    socketio.emit('ice_candidate', {
                        'candidate': data['candidate'],
                        'from': users[request.sid]['id']
                    }, room=viewer_sid)
        else:
            # Send to host
            host_sid = rooms[room_id]['host']
            socketio.emit('ice_candidate', {
                'candidate': data['candidate'],
                'from': users[request.sid]['id']
            }, room=host_sid)

@socketio.on('get_room_info')
def handle_get_room_info(data):
    room_id = data['roomId']
    if room_id in rooms:
        emit('room_info', {
            'exists': True,
            'viewerCount': len(rooms[room_id]['viewers'])
        })
    else:
        emit('room_info', {'exists': False})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)