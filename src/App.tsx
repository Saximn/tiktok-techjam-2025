import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home"
import Enrollment from "./pages/Enrollment";
import Streaming from "./pages/Streaming";
import Viewer from "./pages/Viewer";

function App() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center">
      <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/enrollment" element={<Enrollment />} />
        <Route path="/streaming" element={<Streaming />} />
        <Route path="/viewer" element={<Viewer />} />
      </Routes>
    </Router>
    </div>
  )
}

export default App