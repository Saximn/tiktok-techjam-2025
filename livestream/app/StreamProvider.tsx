// app/StreamProvider.tsx
"use client";

import { StreamTheme } from "@stream-io/video-react-sdk";
import "@stream-io/video-react-sdk/dist/css/styles.css";

export function StreamProvider({ children }: { children: React.ReactNode }) {
  return (
    <StreamTheme style={{ fontFamily: "sans-serif", color: "white" }}>
      {children}
    </StreamTheme>
  );
}
