"use client";

import { useEffect, useState } from "react";
import { StreamVideoClient, StreamCall, useCall, useCallStateHooks, ParticipantView, User } from "@stream-io/video-react-sdk";

const apiKey = "mmhfdzb5evj2";
const token =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL3Byb250by5nZXRzdHJlYW0uaW8iLCJzdWIiOiJ1c2VyL1NhdGluX0ZsdW9yaW5lIiwidXNlcl9pZCI6IlNhdGluX0ZsdW9yaW5lIiwidmFsaWRpdHlfaW5fc2Vjb25kcyI6NjA0ODAwLCJpYXQiOjE3NTYzNjE5MzMsImV4cCI6MTc1Njk2NjczM30.gIbHFlKXPSEf5n3CAm4Ol4epII3A_53ZPCybgX3UTyk";
const user: User = { id: "Satin_Fluorine", name: "Tutorial" };
const callId = "u7hhp9CB9bhEovc5zlb0N";

export default function App() {
  const [call, setCall] = useState<any>(null);

  useEffect(() => {
    const client = new StreamVideoClient({ apiKey, user, token });
    const newCall = client.call("livestream", callId);
    newCall.join({ create: true }).catch(console.error);
    setCall(newCall);

    return () => {
      newCall.leave().catch(console.error);
    };
  }, []);

  if (!call) return null;

  return (
    <StreamCall call={call}>
      <LivestreamView />
    </StreamCall>
  );
}

const LivestreamView = () => {
  const call = useCall();
  const {
    useCameraState,
    useMicrophoneState,
    useParticipantCount,
    useIsCallLive,
    useParticipants,
    useLocalParticipant
  } = useCallStateHooks();

  const localParticipant = useLocalParticipant();
  const [firstParticipant] = useParticipants();
  const participantCount = useParticipantCount();
  const isLive = useIsCallLive();

  // Only access camera/mic if local participant exists
  const camState = useCameraState();
  const micState = useMicrophoneState();

  if (!call) return <div>Joining...</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div>{isLive ? `Live: ${participantCount}` : `In Backstage`}</div>

      {firstParticipant ? (
        <ParticipantView participant={firstParticipant} />
      ) : (
        <div>The host hasn't joined yet</div>
      )}

      {/* Only render controls if local participant exists */}
      {localParticipant && camState.camera && micState.microphone && (
        <div style={{ display: "flex", gap: "8px" }}>
          <button onClick={() => (isLive ? call.stopLive() : call.goLive())}>
            {isLive ? "Stop Live" : "Go Live"}
          </button>
          <button onClick={() => camState.camera.toggle()}>
            {camState.isEnabled ? "Disable camera" : "Enable camera"}
          </button>
          <button onClick={() => micState.microphone.toggle()}>
            {micState.isEnabled ? "Mute Mic" : "Unmute Mic"}
          </button>
        </div>
      )}
    </div>
  );
};

