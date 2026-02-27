import axios from 'axios';
import { Platform } from 'react-native';

// ── ngrok tunnel URL ──────────────────────────────────────────────────────────
// Update this URL every time you restart ngrok (free tier generates a new URL).
const NGROK_URL = 'https://discrepantly-tightknit-julienne.ngrok-free.dev';

const getBaseUrl = () => {
  return NGROK_URL;
};

const API = axios.create({
  baseURL: getBaseUrl(),
  timeout: 10000, // 10 seconds timeout
  headers: {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
  },
});

export const analyzeContent = async (text) => {
  try {
    // Replace '/api/analyze' with your actual Python backend endpoint
    // We are mocking a response here for now to ensure the UI works before the IP is set correctly.
    
    // Call the actual Python backend endpoint
    console.log(`Sending request to: ${getBaseUrl()}/analyze_text`);
    const response = await API.post('/analyze_text', { content: text });
    console.log('response', response)
    return response.data;

  } catch (error) {
    console.error('API Error:', error.message);
    throw error;
  }
};

export const analyzeAudio = async (uri, name, type) => {
  try {
    const formData = new FormData();
    formData.append('file', {
      uri: Platform.OS === 'android' ? uri : uri.replace('file://', ''),
      name: name || 'audio.m4a',
      type: type || 'audio/m4a',
    });

    console.log(`Uploading audio to: ${getBaseUrl()}/analyze_audio`);
    
    // The backend expects multipart/form-data for file uploads
    const response = await API.post('/analyze_audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // Increase timeout for audio upload and processing (which takes longer due to ML/transcription)
      timeout: 30000, 
    });
    
    return response.data;
  } catch (error) {
    console.error('Audio Upload Error:', error.message);
    throw error;
  }
};
