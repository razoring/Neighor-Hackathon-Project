import React, { useState, useEffect, useRef } from 'react';
import { useReactMediaRecorder } from 'react-media-recorder';
import axios from 'axios';
import './App.css'; // Ensure Tailwind directives are here

const App = () => {
  const [sessionId] = useState(`sess_${Math.floor(Math.random() * 10000)}`);
  const [messages, setMessages] = useState([]); // { sender: 'bot' | 'user', text: '' }
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [diagnosis, setDiagnosis] = useState(null);
  const [botAudioSrc, setBotAudioSrc] = useState(null);
  const sentAudioRef = useRef(null);

  const {
    startRecording,
    stopRecording,
    mediaBlobUrl,
    status
  } = useReactMediaRecorder({ audio: true });

  // When recording stops, send audio to backend (only once)
  useEffect(() => {
    if (status === 'stopped' && mediaBlobUrl && sentAudioRef.current !== mediaBlobUrl) {
      sentAudioRef.current = mediaBlobUrl;
      sendAudioToBackend(mediaBlobUrl);
    }
  }, [mediaBlobUrl, status]);

  const sendAudioToBackend = async (url) => {
    const audioBlob = await fetch(url).then(r => r.blob());
    const formData = new FormData();
    formData.append('file', audioBlob, 'input.wav');
    formData.append('session_id', sessionId);
    formData.append('history', JSON.stringify(messages));

    // Add user message placeholder (Transcription would update this)
    setMessages(prev => [...prev, { sender: 'user', text: '(Audio Sent)' }]);

    try {
      const res = await axios.post('http://localhost:8000/chat', formData);
      const { response_text, audio_base64 } = res.data;

      // Update Chat
      setMessages(prev => [...prev, { sender: 'bot', text: response_text }]);

      // Play Audio (Assuming backend returns text, we fetch audio separately or handle base64)
      // For this demo code, let's assume we fetch audio by text
      const audioRes = await axios.get(`http://localhost:8000/audio_response?text=${encodeURIComponent(response_text)}`, { responseType: 'blob' });
      const audioUrl = URL.createObjectURL(audioRes.data);
      const audio = new Audio(audioUrl);
      audio.play();

    } catch (err) {
      console.error(err);
    }
  };

  const handleEndCall = async () => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);
      const res = await axios.post('http://localhost:8000/analyze', formData);
      const data = res.data;

      // The backend returns { local_inference, gemini_explanation }
      const local = data.local_inference || {};
      const gemini = data.gemini_explanation || {};

      // Prefer structured Gemini explanation if present, otherwise use local inference
      const merged = {
        label: gemini.label || local.label || 'Unknown',
        score: gemini.score ?? local.score ?? 0,
        confidence: gemini.confidence || local.confidence || 'Low',
        explanation: gemini.explanation || gemini.explanation_text || `Local heuristic: ${JSON.stringify(local)}`,
        raw: data,
      };

      setDiagnosis(merged);
    } catch (err) {
      console.error("Analysis failed", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center font-sans">
      <div className="w-full max-w-md bg-gray-800 rounded-2xl shadow-2xl overflow-hidden">
        
        {/* Header */}
        <div className="bg-indigo-600 p-4 text-center">
          <h1 className="text-xl font-bold">Memory Check Companion</h1>
          <p className="text-indigo-200 text-sm">Conversational AI Assessment</p>
        </div>

        {/* Chat Area */}
        <div className="h-96 overflow-y-auto p-4 space-y-4 bg-gray-900">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-xs p-3 rounded-lg ${msg.sender === 'user' ? 'bg-indigo-500' : 'bg-gray-700'}`}>
                {msg.text}
              </div>
            </div>
          ))}
          {status === 'recording' && <p className="text-center text-red-400 animate-pulse">Listening...</p>}
        </div>

        {/* Controls */}
        <div className="p-4 bg-gray-800 flex justify-center gap-4 border-t border-gray-700">
          {!diagnosis ? (
            <>
              <button 
                onMouseDown={startRecording} 
                onMouseUp={stopRecording}
                className="bg-green-500 hover:bg-green-600 text-white w-16 h-16 rounded-full flex items-center justify-center shadow-lg transition transform hover:scale-105"
              >
                🎤
              </button>
              <button 
                onClick={handleEndCall}
                className="bg-red-500 hover:bg-red-600 text-white px-6 py-2 rounded-full shadow-lg font-bold"
              >
                End Call & Analyze
              </button>
            </>
          ) : (
            <div className="w-full text-center">
              <h2 className="text-2xl font-bold mb-2">Results</h2>
              <div className="bg-gray-700 p-4 rounded-lg text-left">
                <p><span className="text-indigo-400 font-bold">Diagnosis:</span> {diagnosis.label}</p>
                <p><span className="text-indigo-400 font-bold">Score:</span> {diagnosis.score}/100</p>
                <p><span className="text-indigo-400 font-bold">Confidence:</span> {diagnosis.confidence}</p>
                <p className="mt-2 text-sm text-gray-300">{diagnosis.explanation}</p>
              </div>
              <button onClick={() => window.location.reload()} className="mt-4 text-gray-400 underline">Start Over</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;