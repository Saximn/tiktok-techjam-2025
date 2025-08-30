### Install packages

```bash
# Backend
pip install fastapi uvicorn[standard] aiortc opencv-python mediapipe av numpy
```

Run the Demo

Backend

```bash
uvicorn vision_guard_server:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile ./server.key --ssl-certfile ./server.crt --reload
```