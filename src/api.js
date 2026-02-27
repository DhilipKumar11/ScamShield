import axios from "axios";

const API_BASE = "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

export async function analyzeAudio(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/analyze_audio", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function sendLiveChunk(sessionId, blob) {
  const formData = new FormData();
  formData.append("file", blob, "chunk.webm");
  formData.append("session_id", sessionId);
  const response = await api.post("/live_chunk", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function getReport(sessionId) {
  const response = await api.get(`/report/${sessionId}`);
  return response.data;
}
