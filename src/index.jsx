import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './ui/App.jsx';

// Initialize React app
const container = document.getElementById('root');
const root = createRoot(container);

// Error boundary for development
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    
    componentDidCatch(error, errorInfo) {
        console.error('VoiceShield Error:', error, errorInfo);
    }
    
    render() {
        if (this.state.hasError) {
            return (
                <div className="error-boundary">
                    <h1>🚨 VoiceShield Error</h1>
                    <p>Something went wrong with the VoiceShield application.</p>
                    <details>
                        <summary>Error Details</summary>
                        <pre>{this.state.error?.toString()}</pre>
                    </details>
                    <button onClick={() => window.location.reload()}>
                        Reload Application
                    </button>
                </div>
            );
        }
        
        return this.props.children;
    }
}

// Render the app with error boundary
root.render(
    <ErrorBoundary>
        <App />
    </ErrorBoundary>
);

// Log VoiceShield initialization
console.log('🛡️ VoiceShield initialized successfully');
console.log('🚀 Ready for real-time voice privacy protection');

// Service worker registration for PWA functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
