import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home"
import Enrollment from "./pages/Enrollment";
import Streaming from "./pages/Streaming";

function App() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center">
      <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/enrollment" element={<Enrollment />} />
        <Route path="/streaming" element={<Streaming />} />
      </Routes>
    </Router>
    </div>
  )
}

export default App