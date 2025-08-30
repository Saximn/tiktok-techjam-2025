import { useState } from "react";
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  ParticipantView,
  useCallStateHooks,
} from "@stream-io/video-react-sdk";
import type { User } from "@stream-io/video-react-sdk";
import { Button } from "@/components/ui/button";

const apiKey = "mmhfdzb5evj2";
const token =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL3Byb250by5nZXRzdHJlYW0uaW8iLCJzdWIiOiJ1c2VyL1F1aWNrZXN0X0dob3VsIiwidXNlcl9pZCI6IlF1aWNrZXN0X0dob3VsIiwidmFsaWRpdHlfaW5fc2Vjb25kcyI6NjA0ODAwLCJpYXQiOjE3NTY0MDA1NzEsImV4cCI6MTc1NzAwNTM3MX0.MnJzcI1K8E9-sq-LXvdgoTY2P_LzUd2xAqbfy7NRjlk";
const userId = "Quickest_Ghoul";
const user: User = { id: userId, name: "Viewer" };

export default function Viewer() {
  const [callId, setCallId] = useState("");
  const [call, setCall] = useState<ReturnType<
    typeof StreamVideoClient.prototype.call
  > | null>(null);
  const [client, setClient] = useState<StreamVideoClient | null>(null);
  const [joined, setJoined] = useState(false);

  const handleJoin = async () => {
    try {
      const streamClient = new StreamVideoClient({ apiKey, user, token });
      const streamCall = streamClient.call("livestream", callId);

      await streamCall.camera.disable();
      await streamCall.microphone.disable();

      await streamCall.join({ create: false });
      setClient(streamClient);
      setCall(streamCall);
      setJoined(true);
    } catch (error) {
      console.error("Failed to join stream:", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen w-screen bg-black text-white">
      {!joined ? (
        <div className="w-full max-w-md flex flex-col gap-6 p-6">
          <h1 className="text-3xl font-bold text-center">Join Live Stream</h1>
          <input
            className="w-full px-4 py-2 rounded-md bg-black text-white border border-white focus:outline-none"
            type="text"
            placeholder="Enter Stream ID"
            value={callId}
            onChange={(e) => setCallId(e.target.value)}
          />
          <Button
            className="w-full bg-white text-black border border-white hover:bg-black hover:text-white transition"
            onClick={handleJoin}
          >
            Join Stream
          </Button>
        </div>
      ) : (
        client &&
        call && (
          <StreamVideo client={client}>
            <StreamCall call={call}>
              <LivestreamView />
            </StreamCall>
          </StreamVideo>
        )
      )}
    </div>
  );
}

const LivestreamView = () => {
  const { useParticipantCount, useIsCallLive, useParticipants } =
    useCallStateHooks();

  const participantCount = useParticipantCount();
  const isLive = useIsCallLive();
  const [firstParticipant] = useParticipants();

  return (
    <div className="flex flex-col w-full h-full bg-black text-white">
      {/* Header Bar */}
      <div className="w-full flex items-center justify-between px-6 py-3 border-b border-white">
        <h2 className="text-lg font-semibold">
          {isLive ? "● Live" : "Offline"}
        </h2>
        <span className="text-sm">{participantCount} Watching</span>
      </div>

      {/* Video Section */}
      <div className="flex flex-1 items-center justify-center p-6">
        <div className="w-full max-w-6xl aspect-video border border-white rounded-lg flex items-center justify-center bg-black">
          {isLive && firstParticipant ? (
            <ParticipantView participant={firstParticipant} />
          ) : (
            <span className="text-gray-400">
              {isLive
                ? "The host hasn't joined yet."
                : "The livestream has not started."}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
