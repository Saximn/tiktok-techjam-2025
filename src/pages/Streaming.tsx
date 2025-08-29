import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
} from "@stream-io/video-react-sdk";
import type { User } from "@stream-io/video-react-sdk";
import { ParticipantView, useCallStateHooks } from "@stream-io/video-react-sdk";
import { Button } from "@/components/ui/button"


const apiKey = "mmhfdzb5evj2";
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL3Byb250by5nZXRzdHJlYW0uaW8iLCJzdWIiOiJ1c2VyL1F1aWNrZXN0X0dob3VsIiwidXNlcl9pZCI6IlF1aWNrZXN0X0dob3VsIiwidmFsaWRpdHlfaW5fc2Vjb25kcyI6NjA0ODAwLCJpYXQiOjE3NTY0MDA1NzEsImV4cCI6MTc1NzAwNTM3MX0.MnJzcI1K8E9-sq-LXvdgoTY2P_LzUd2xAqbfy7NRjlk";
const userId = "Quickest_Ghoul";
const callId = "HdPgrADAhosrXNQ5wEaH9";

const user: User = { id: userId, name: "Tutorial" };
const client = new StreamVideoClient({ apiKey, user, token });
const call = client.call("livestream", callId);
call.join({ create: true });

export default function Streaming() {
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
    <div className="flex flex-col gap-4 min-h-screen bg-white dark:bg-black p-4">
  {/* Live / Backstage Status */}
  <div className="text-lg font-medium text-black dark:text-white">
    {isLive ? `Live: ${participantCount}` : `In Backstage`}
  </div>

  {/* Participant View */}
  {firstParticipant ? (
    <ParticipantView participant={firstParticipant} />
  ) : (
    <div className="text-gray-600 dark:text-gray-400">
      The host hasn't joined yet
    </div>
  )}

  {/* Controls */}
  <div className="flex gap-4">
    <Button
      className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
      onClick={() => (isLive ? call.stopLive() : call.goLive())}
    >
      {isLive ? "Stop Live" : "Go Live"}
    </Button>
    <Button
      className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
      onClick={() => cam.toggle()}
    >
      {isCamEnabled ? "Disable camera" : "Enable camera"}
    </Button>
    <Button
      className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
      onClick={() => mic.toggle()}
    >
      {isMicEnabled ? "Mute Mic" : "Unmute Mic"}
    </Button>
  </div>
</div>

  );
};