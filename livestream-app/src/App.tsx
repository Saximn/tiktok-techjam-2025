import {
  LivestreamPlayer,
  StreamVideo,
  StreamVideoClient,
  User,
  StreamCall,
} from "@stream-io/video-react-sdk";

const apiKey = "xvzfdg7f9d7m";
const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJAc3RyZWFtLWlvL2Rhc2hib2FyZCIsImlhdCI6MTc1NjMxMDAzNiwiZXhwIjoxNzU2Mzk2NDM2LCJ1c2VyX2lkIjoiIWFub24iLCJyb2xlIjoidmlld2VyIiwiY2FsbF9jaWRzIjpbImxpdmVzdHJlYW06bGl2ZXN0cmVhbV9hYTI4MWJiOS1lMGFkLTQwMGYtYTA5Yi03OTViNzM3YWQzOTkiXX0.pWUwOB1ZZmR-p7k59VXzLLM32Zq_3dfzQ8Oi_HqDwoc";
const callId = "livestream_aa281bb9-e0ad-400f-a09b-795b737ad399";

const user: User = { type: "anonymous" };
const client = new StreamVideoClient({ apiKey, user, token });
const call = client.call("livestream", callId);

export default function App() {
  return (
    <StreamVideo client={client}>
      <LivestreamPlayer callType="livestream" callId={callId} />
    </StreamVideo>
  );
}