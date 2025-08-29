import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import './index.css'
import { StreamTheme } from "@stream-io/video-react-sdk";
import { StrictMode } from "react";

// import the SDK provided styles
import "@stream-io/video-react-sdk/dist/css/styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <StrictMode>
  <StreamTheme style={{ fontFamily: "sans-serif", color: "white" }}>
    <App />
  </StreamTheme>
  </StrictMode>
);