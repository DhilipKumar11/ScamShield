import React, { useState, useEffect } from 'react';
import UploadMode from './components/UploadMode';
import LiveMode from './components/LiveMode';

function App() {
  const [mode, setMode] = useState('upload');
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('scamshield-theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('scamshield-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">🛡️</div>
          <span className="header-title">Voice Scam & AI Call Risk Detector</span>
          <span className="system-badge">
            <span className="dot"></span>
            System Active
          </span>
        </div>
        <div className="header-right">
          <nav className="header-nav">
            <button className="nav-link" onClick={() => alert('Documentation: ScamShield uses AI-powered speech analysis to detect voice scams and synthetic voice manipulation in real-time.')}>Documentation</button>
            <button className="nav-link" onClick={() => alert('History feature coming soon! Session reports are saved locally when you click Report.')}>History</button>
          </nav>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
          <button className="settings-btn" title="Settings" onClick={() => alert('Settings: Configure detection sensitivity, language preferences, and notification options.')}>
            ⚙️
          </button>
          <div className="avatar" title="User Profile">U</div>
        </div>
      </header>

      {/* Mode Toggle */}
      <div className="mode-toggle-container">
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
            onClick={() => setMode('upload')}
          >
            <span className="icon">📤</span>
            Upload Audio
          </button>
          <button
            className={`mode-btn ${mode === 'live' ? 'active' : ''}`}
            onClick={() => setMode('live')}
          >
            <span className="icon">🎙️</span>
            Live Voice Mode
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="main-content">
        {mode === 'upload' ? <UploadMode /> : <LiveMode />}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-left">© 2024 SecureVoice AI Systems</div>
        <div className="footer-right">
          <button className="footer-link" onClick={() => alert('Privacy Policy: All audio is processed locally and deleted after analysis. No data is stored permanently.')}>Privacy Policy</button>
          <button className="footer-link" onClick={() => alert('Terms of Service: ScamShield is provided as-is for educational and protective purposes.')}>Terms of Service</button>
          <button className="footer-link" onClick={() => alert('Support Portal: Contact support@securevoice.ai for assistance.')}>Support Portal</button>
        </div>
      </footer>
    </div>
  );
}

export default App;
