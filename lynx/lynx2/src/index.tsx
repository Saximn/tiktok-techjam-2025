import React from 'react';
import { MemoryRouter, Routes, Route } from 'react-router';
import ReactDOM from 'react-dom/client';
import App from './App';
import Enrollment from './components/enrollment/enrollment';

const rootEl = document.getElementById('root');
if (rootEl) {
  const root = ReactDOM.createRoot(rootEl);
  root.render(
    <React.StrictMode>
      <MemoryRouter>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/enrollment" element={<Enrollment />} />
    </Routes>
  </MemoryRouter>,
    </React.StrictMode>,
  );
}
