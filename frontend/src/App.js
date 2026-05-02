import React, { useState } from 'react';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('landing'); // 'landing', 'analyzer', 'insights'
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      if (data.success) {
        setPrediction(data);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError('Cannot connect to backend. Make sure server is running on port 8000');
    } finally {
      setLoading(false);
    }
  };

  // Helper to render bold markdown
  const renderBoldText = (text) => {
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index} className="highlight-text">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  return (
    <div className="app">
      {/* Animated Bubbles - Small and numerous */}
      <div className="bubbles">
        {[...Array(100)].map((_, i) => (
          <div key={i} className="bubble" style={{ 
            left: `${Math.random() * 100}%`,
            width: `${Math.random() * 20 + 5}px`,
            height: `${Math.random() * 20 + 5}px`,
            animationDuration: `${Math.random() * 15 + 10}s`,
            animationDelay: `${Math.random() * 10}s`,
            opacity: Math.random() * 0.4 + 0.1
          }}></div>
        ))}
      </div>

      {currentView === 'landing' && (
        <div className="landing-view fade-in">
          <div className="globe-container">
            <div className="globe"></div>
          </div>
          <h1 className="landing-title">AI Satellite Analyzer</h1>
          <p className="landing-subtitle">Advanced Earth Observation | Hybrid CNN + Gemini AI</p>
          <button className="start-btn" onClick={() => setCurrentView('analyzer')}>
            INITIALIZE UPLINK
          </button>
        </div>
      )}

      {currentView !== 'landing' && (
        <header className="corner-header fade-in" onClick={() => setCurrentView('landing')} title="Return to Home">
          <h1>🛰️ AI Analyzer</h1>
          <div className="badge-group">
            <span className="badge">✨ Hybrid AI</span>
          </div>
        </header>
      )}

      {currentView === 'analyzer' && (
        <div className="container slide-up">
          <div className="card analyzer-card">
            <div className="upload-area">
              <label className="upload-label">
                <span className="icon">📸</span> Choose Satellite Image
                <input type="file" accept="image/*" onChange={handleFileChange} hidden />
              </label>

              {preview && (
                <div className="preview-container fade-in">
                  <img src={preview} alt="Satellite Preview" className="preview-image" />
                  <button className="analyze-btn" onClick={handleAnalyze} disabled={loading}>
                    {loading ? '🔍 Scanning Data...' : '🚀 Analyze Image'}
                  </button>
                </div>
              )}

              {loading && (
                <div className="loading">
                  <div className="spinner"></div>
                  <p>🤖 AI is scanning topographical features...</p>
                </div>
              )}

              {error && <div className="error">{error}</div>}
            </div>

            {prediction && (
              <div className="results fade-in">
                <div className="main-prediction">
                  <div className="prediction-icon">🎯</div>
                  <div className="prediction-label">Primary Classification</div>
                  <div className="prediction-class">{prediction.classification}</div>
                  <div className="confidence-bar-container">
                    <div className="confidence-bar">
                      <div className="confidence-fill" style={{ width: `${prediction.confidence}%` }}></div>
                    </div>
                    <div className="confidence-text">Confidence: {prediction.confidence}%</div>
                  </div>
                </div>

                <div className="top-predictions">
                  <h3>📊 Alternative Classifications</h3>
                  {prediction.top_predictions.map((item, idx) => (
                    <div key={idx} className="prediction-bar">
                      <div className="bar-label">
                        <span>{idx + 1}. {item.class}</span>
                        <span>{item.confidence}%</span>
                      </div>
                      <div className="bar-bg">
                        <div className="bar-fill" style={{ width: `${item.confidence}%` }}>
                          {item.confidence > 20 && `${item.confidence}%`}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <button className="view-insights-btn" onClick={() => setCurrentView('insights')}>
                  💡 View Deep AI Insights
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {currentView === 'insights' && prediction && (
        <div className="container slide-up">
          <div className="card insights-page">
            <button className="back-btn" onClick={() => setCurrentView('analyzer')}>
              ← Back to Analysis
            </button>
            <div className="insights-header">
              <div className="insights-icon-large">💡</div>
              <h2>AI Insights & Recommendations</h2>
            </div>
            <div className="insights-content">
              {prediction.analysis.split('\n').map((line, idx) => (
                <p key={idx}>{renderBoldText(line)}</p>
              ))}
            </div>
            <div className="source-badge">
              {prediction.gemini_used ? '🤖 Enhanced with Google Gemini AI' : '🧠 Powered by Custom CNN'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;