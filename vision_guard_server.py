# vision_guard_server.py
import cv2
import av
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import itertools
from fractions import Fraction

app = FastAPI()

# Initialize MediaPipe Face Detector in VIDEO mode
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blazeface_short_range.tflite'),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5
)

face_detector = FaceDetector.create_from_options(options)

# Video track processor
class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.last_timestamp_ms = -1

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Compute timestamp in milliseconds
        timestamp_ms = int(frame.time * 1000)
        # Ensure monotonic increase
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        try:
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

            detections = face_detector.detect_for_video(mp_image, timestamp_ms)

            

            # Blur mouth keypoints for all faces
            h, w, _ = img.shape
            for det in detections.detections:
                mouth = det.keypoints[3]  # 0:left_eye, 1:right_eye, 2:nose, 3:mouth
                mx, my = int(mouth.x * w), int(mouth.y * h)
                size = 60
                x1, y1 = max(0, mx - size), max(0, my - size)
                x2, y2 = min(w, mx + size), min(h, my + size)
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    img[y1:y2, x1:x2] = roi

        except Exception as e:
            print("Face detection error:", e)

        # Correct PTS / time_base
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# Simple HTML UI
@app.get("/")
async def index():
    return HTMLResponse("""
    <html>
    <body>
        <h2>Server-side Mouth Blur Test</h2>
        <video id="localVideo" autoplay muted playsinline></video>
        <video id="processedVideo" autoplay playsinline></video>
        <button onclick="start()">Start</button>
        <script>
        async function start() {
            const pc = new RTCPeerConnection();
            const localStream = await navigator.mediaDevices.getUserMedia({ video: true });
            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
            console.log(localStream.getVideoTracks());
            document.getElementById("localVideo").srcObject = localStream;

           pc.ontrack = (event) => {
                const stream = event.streams[0];
                const videoEl = document.getElementById("processedVideo");
                videoEl.srcObject = stream;
                videoEl.play();  // explicitly play
            };

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            const ws = new WebSocket("wss://" + window.location.host + "/offer");
            ws.onopen = () => ws.send(JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }));
            ws.onmessage = async (msg) => {
                const data = JSON.parse(msg.data);
                await pc.setRemoteDescription(data);
            };
        }
        </script>
    </body>
    </html>
    """)

@app.websocket("/offer")
async def offer(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_json()
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        if track.kind == "video":
            processed_track = ProcessedVideoTrack(track)
            pc.addTrack(processed_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await ws.send_json({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

