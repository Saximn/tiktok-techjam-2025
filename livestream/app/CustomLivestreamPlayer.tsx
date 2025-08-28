"use client";

import { useEffect, useState } from "react";
import {
  Call,
  ParticipantView,
  StreamCall,
  useCallStateHooks,
  useStreamVideoClient,
} from "@stream-io/video-react-sdk";

export const CustomLivestreamPlayer = (props: { callType: string; callId: string }) => {
  const { callType, callId } = props;
  const client = useStreamVideoClient();
  const [call, setCall] = useState<Call>();

  useEffect(() => {
    if (!client) return;

    const myCall = client.call(callType, callId);
    myCall.join({ create: true }).catch((e) => console.error("Failed to join call", e));
    setCall(myCall);

    return () => {
      myCall.leave().catch((e) => console.error("Failed to leave call", e));
      setCall(undefined);
    };
  }, [client, callType, callId]);

  if (!call) return null;

  return (
    <StreamCall call={call}>
      <CustomLivestreamLayout />
    </StreamCall>
  );
};

const CustomLivestreamLayout = () => {
  const { useParticipants, useParticipantCount } = useCallStateHooks();
  const participantCount = useParticipantCount();
  const [firstParticipant] = useParticipants();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <div>Live: {participantCount}</div>
      {firstParticipant ? <ParticipantView participant={firstParticipant} /> : <div>The host hasn't joined yet</div>}
    </div>
  );
};
