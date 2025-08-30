# Live Streaming App

A simple live streaming application built with Next.js frontend and Flask backend, designed to serve as a foundation for implementing streaming filters.

## Features

- **Host Interface**: Start streaming with webcam and microphone
- **Viewer Interface**: Join streams using room IDs
- **Real-time Communication**: WebRTC peer-to-peer connections
- **WebSocket Signaling**: Flask-SocketIO for connection management
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

- **Frontend**: Next.js with TypeScript, Tailwind CSS
- **Backend**: Flask with Socket.IO for WebRTC signaling
- **Communication**: WebRTC for media streaming, WebSocket for signaling

## Setup Instructions

### Backend (Flask)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```

The backend will start on `http://localhost:5000`

### Frontend (Next.js)

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

The frontend will start on `http://localhost:3000`

## Usage

### Starting a Stream (Host)

1. Open `http://localhost:3000`
2. Click "Start Streaming"
3. Allow camera and microphone permissions
4. Click "Start Streaming" button
5. Share the generated Room ID with viewers

### Joining a Stream (Viewer)

1. Open `http://localhost:3000`
2. Enter the Room ID provided by the host
3. Click "Join Stream"
4. Wait for the stream to connect

## Project Structure

```
├── backend/
│   ├── app.py              # Flask server with Socket.IO
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Home page
│   │   ├── host/
│   │   │   └── page.tsx    # Host streaming interface
│   │   ├── viewer/
│   │   │   └── [roomId]/
│   │   │       └── page.tsx # Viewer interface
│   │   ├── layout.tsx      # App layout
│   │   └── globals.css     # Global styles
│   ├── lib/
│   │   ├── webrtc.ts       # WebRTC utilities
│   │   └── socket.ts       # Socket.IO client
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── tsconfig.json
└── README.md
```

## Technical Details

### WebRTC Flow

1. Host creates a room and starts streaming
2. Viewer joins the room using Room ID
3. Backend facilitates WebRTC signaling via Socket.IO
4. Peer-to-peer connection established for media streaming
5. STUN servers used for NAT traversal

### Socket.IO Events

- `create_room`: Host creates a new streaming room
- `join_room`: Viewer joins an existing room
- `offer/answer`: WebRTC session negotiation
- `ice_candidate`: ICE candidate exchange
- `viewer_joined/left`: Room participant tracking

## Future Enhancements

This app serves as a foundation for implementing streaming filters such as:

- Sensitive data blurring
- Mouth blurring for sensitive words
- Audio beeping/censoring
- Real-time content analysis
- Custom filtering rules

The filtering logic can be implemented server-side in the Flask backend to process the media streams before forwarding to viewers.

## Troubleshooting

### Common Issues

1. **Camera/Microphone Access**: Ensure browser permissions are granted
2. **HTTPS Required**: WebRTC requires HTTPS in production
3. **Firewall/NAT**: May need TURN servers for complex network setups
4. **Browser Compatibility**: Use modern browsers with WebRTC support

### Development Tips

- Use browser developer tools to debug WebRTC connections
- Check Flask server logs for Socket.IO events
- Monitor network tab for WebSocket connections
- Test with multiple browser tabs/windows