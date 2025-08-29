import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
} from "@stream-io/video-react-sdk";
import type { User } from "@stream-io/video-react-sdk";
import { ParticipantView, useCallStateHooks } from "@stream-io/video-react-sdk";


const apiKey = "mmhfdzb5evj2";
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL3Byb250by5nZXRzdHJlYW0uaW8iLCJzdWIiOiJ1c2VyL1F1aWNrZXN0X0dob3VsIiwidXNlcl9pZCI6IlF1aWNrZXN0X0dob3VsIiwidmFsaWRpdHlfaW5fc2Vjb25kcyI6NjA0ODAwLCJpYXQiOjE3NTY0MDA1NzEsImV4cCI6MTc1NzAwNTM3MX0.MnJzcI1K8E9-sq-LXvdgoTY2P_LzUd2xAqbfy7NRjlk";
const userId = "Quickest_Ghoul";
const callId = "HdPgrADAhosrXNQ5wEaH9";

const user: User = { id: userId, name: "Tutorial" };
const client = new StreamVideoClient({ apiKey, user, token });
const call = client.call("livestream", callId);
call.join({ create: true });

export default function App() {
  return (
    <StreamVideo client={client}>
      <StreamCall call={call}>
        <LivestreamView />
      </StreamCall>
    </StreamVideo>
  );
}

const LivestreamView = () => {
  const {
    useCameraState,
    useMicrophoneState,
    useParticipantCount,
    useIsCallLive,
    useParticipants,
  } = useCallStateHooks();

  const { camera: cam, isEnabled: isCamEnabled } = useCameraState();
  const { microphone: mic, isEnabled: isMicEnabled } = useMicrophoneState();

  const participantCount = useParticipantCount();
  const isLive = useIsCallLive();

  const [firstParticipant] = useParticipants();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <div>{isLive ? `Live: ${participantCount}` : `In Backstage`}</div>
      {firstParticipant ? (
        <ParticipantView participant={firstParticipant} />
      ) : (
        <div>The host hasn't joined yet</div>
      )}
      <div style={{ display: "flex", gap: "4px" }}>
        <button onClick={() => (isLive ? call.stopLive() : call.goLive())}>
          {isLive ? "Stop Live" : "Go Live"}
        </button>
        <button onClick={() => cam.toggle()}>
          {isCamEnabled ? "Disable camera" : "Enable camera"}
        </button>
        <button onClick={() => mic.toggle()}>
          {isMicEnabled ? "Mute Mic" : "Unmute Mic"}
        </button>
      </div>
    </div>
  );
};