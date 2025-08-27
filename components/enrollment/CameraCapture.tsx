"use client";

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

/**
 * CameraCapture component
 * Provides camera access and image preview for whitelist enrollment.
 */
export function CameraCapture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);

  const openCamera = async () => {
    setIsCameraOpen(true);
    if (videoRef.current) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        setImageSrc(canvas.toDataURL("image/png"));
      }
    }
  };

  const closeCamera = () => {
    setIsCameraOpen(false);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  return (
    <Card className="mb-4 p-4">
      {!isCameraOpen ? (
        <Button variant="outline" onClick={openCamera}>
          Open Camera
        </Button>
      ) : (
        <div className="flex flex-col items-center">
          <video
            ref={videoRef}
            autoPlay
            className="rounded mb-2 w-full max-w-xs"
          />
          <div className="flex gap-2">
            <Button onClick={captureImage}>Capture</Button>
            <Button variant="destructive" onClick={closeCamera}>
              Close
            </Button>
          </div>
        </div>
      )}
      <canvas ref={canvasRef} style={{ display: "none" }} />
      {imageSrc && (
        <div className="mt-4 flex flex-col items-center">
          <span className="text-muted-foreground mb-2">Captured Image:</span>
          <img
            src={imageSrc}
            alt="Captured"
            className="rounded shadow w-full max-w-xs"
          />
        </div>
      )}
    </Card>
  );
}
