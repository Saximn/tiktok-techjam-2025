import './App.css';
import '@lynx-js/web-core/index.css';
import '@lynx-js/web-elements/index.css';
import '@lynx-js/web-core';
import '@lynx-js/web-elements/all';
import { useNavigate } from 'react-router';

const App = () => {
  const nav = useNavigate();
  return (
    <div className="content">
      <text>LPS Control</text>
      <text onClick={() => nav('/enrollment')}>Set Approved Streamer</text>
    </div>
  );
};

export default App;
