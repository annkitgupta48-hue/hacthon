const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const messageOutput = document.getElementById('messageOutput');
const handValue = document.getElementById('handValue');
const faceValue = document.getElementById('faceValue');
const gestureConfidenceEl = document.getElementById('gestureConfidence');
const expressionConfidenceEl = document.getElementById('expressionConfidence');
const intentValue = document.getElementById('intentValue');
const fpsValue = document.getElementById('fpsValue');
const processingState = document.getElementById('processingState');
const cameraStatus = document.getElementById('cameraStatus');
const historyList = document.getElementById('historyList');
const API_BASE_URL = window.GESTURESPEAK_API_URL || 'http://127.0.0.1:8010';
const frameCanvas = document.createElement('canvas');
const frameContext = frameCanvas.getContext('2d', { willReadFrequently: true });

const state = {
  stream: null,
  isCameraOn: false,
  demoMode: false,
  analysisTimer: null,
  analysisInFlight: false,
  lastAnalysis: null,
  stableText: '',
  stableCount: 0,
  lastEmittedText: '',
  lastEmittedAt: 0,
  composedMessage: '',
  lastEventKey: '',
  recentMessages: JSON.parse(localStorage.getItem('gestureMessages') || '[]'),
};

const demoResponses = {
  thumbs_up: { gesture: 'thumbs_up', expression: 'happy', text: "Yes, that's good!" },
  thumbs_down: { gesture: 'thumbs_down', expression: 'sad', text: 'No, not good.' },
  wave: { gesture: 'wave', expression: 'happy', text: 'Hello!' },
  stop: { gesture: 'stop', expression: 'angry', text: 'Please stop.' },
  happy: { gesture: 'unknown', expression: 'happy', text: 'Yes, I am happy.' },
  sad: { gesture: 'unknown', expression: 'sad', text: "No, I don't like it." },
  angry: { gesture: 'unknown', expression: 'angry', text: 'I am angry.' },
  surprised: { gesture: 'unknown', expression: 'surprised', text: 'I am surprised.' },
};

function updateHistory() {
  historyList.innerHTML = '';
  state.recentMessages.slice(0, 5).forEach((msg) => {
    const li = document.createElement('li');
    li.textContent = msg;
    historyList.appendChild(li);
  });
}

function addMessageToHistory(message) {
  if (!message) return;
  const trimmed = String(message).trim();
  if (!trimmed) return;
  if (state.recentMessages[state.recentMessages.length - 1] === trimmed) {
    return;
  }
  state.recentMessages.push(trimmed);
  state.recentMessages = state.recentMessages.slice(-5);
  localStorage.setItem('gestureMessages', JSON.stringify(state.recentMessages));
  updateHistory();
}

function setMessage(text) {
  messageOutput.value = text;
  addMessageToHistory(text);
}

function appendDetectedMessage(text, eventKey) {
  const now = Date.now();
  const sameEventCooldown = eventKey === state.lastEventKey && now - state.lastEmittedAt < 1600;
  if (sameEventCooldown) return;
  state.lastEventKey = eventKey;
  state.lastEmittedAt = now;
  state.composedMessage = state.composedMessage ? `${state.composedMessage} ${text}` : text;
  messageOutput.value = state.composedMessage;
  processingState.textContent = 'Message added';
}

function speakText() {
  const text = messageOutput.value.trim();
  if (!text || !('speechSynthesis' in window)) {
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function updateDetectionPanel(data) {
  const gesture = data?.gesture || {};
  const expression = data?.expression || {};
  handValue.textContent = gesture.name || 'Unknown';
  faceValue.textContent = expression.name || 'Unknown';
  gestureConfidenceEl.textContent = `${Math.round((gesture.confidence || 0) * 100)}%`;
  expressionConfidenceEl.textContent = `${Math.round((expression.confidence || 0) * 100)}%`;
  intentValue.textContent = data?.intent?.label || 'Unknown';
}

function applyStableAnalysis(data) {
  updateDetectionPanel(data);
  const text = String(data?.text || 'Gesture unclear. Please try again.');
  const intentConfidence = Number(data?.intent?.confidence || 0);
  const gestureConfidence = Number(data?.gesture?.confidence || 0);
  const isActionable = text !== 'Gesture unclear. Please try again.' && (intentConfidence >= 0.72 || gestureConfidence >= 0.82);
  if (!isActionable) {
    processingState.textContent = 'Tracking movement';
    return;
  }
  if (text === state.stableText) {
    state.stableCount += 1;
  } else {
    state.stableText = text;
    state.stableCount = 1;
  }
  const eventKey = `${data?.gesture?.name || 'unknown'}:${text}`;
  if (state.stableCount >= 2) {
    state.lastEmittedText = eventKey;
    appendDetectedMessage(text, eventKey);
  }
}

async function analyzeCurrentFrame() {
  if (!state.isCameraOn || state.analysisInFlight || video.readyState < 2) return;
  state.analysisInFlight = true;
  frameCanvas.width = video.videoWidth || 640;
  frameCanvas.height = video.videoHeight || 480;
  frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  const frame = frameCanvas.toDataURL('image/jpeg', 0.72);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 2200);
  try {
    const response = await fetch(`${API_BASE_URL}/analyze-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`analysis failed: ${response.status}`);
    }
    const data = await response.json();
    state.lastAnalysis = data;
    applyStableAnalysis(data);
    processingState.textContent = 'Live inference';
  } catch (error) {
    processingState.textContent = error.name === 'AbortError' ? 'Analysis timeout' : 'Backend unavailable';
  } finally {
    window.clearTimeout(timeout);
    state.analysisInFlight = false;
  }
}

function startAnalysisLoop() {
  stopAnalysisLoop();
  state.analysisTimer = window.setInterval(analyzeCurrentFrame, 280);
}

function stopAnalysisLoop() {
  if (state.analysisTimer) {
    window.clearInterval(state.analysisTimer);
    state.analysisTimer = null;
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    state.stream = stream;
    state.isCameraOn = true;
    video.srcObject = stream;
    cameraStatus.textContent = 'Camera online';
    cameraStatus.classList.remove('offline');
    cameraStatus.classList.add('online');
    processingState.textContent = 'Live';
    state.composedMessage = '';
    state.lastEventKey = '';
    state.lastEmittedText = '';
    video.addEventListener('loadedmetadata', startAnalysisLoop, { once: true });
    if (video.readyState >= 1) startAnalysisLoop();
  } catch (error) {
    cameraStatus.textContent = 'Permission denied';
    cameraStatus.classList.remove('online');
    cameraStatus.classList.add('offline');
    processingState.textContent = 'Camera unavailable';
    state.demoMode = true;
  }
}

function stopCamera() {
  stopAnalysisLoop();
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  state.isCameraOn = false;
  video.srcObject = null;
  cameraStatus.textContent = 'Camera offline';
  cameraStatus.classList.remove('online');
  cameraStatus.classList.add('offline');
  processingState.textContent = 'Idle';
  state.analysisInFlight = false;
}

function drawDemoOverlay(text) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#5eead4';
  ctx.lineWidth = 3;
  ctx.strokeRect(40, 40, 560, 400);
  ctx.fillStyle = '#edf2ff';
  ctx.font = 'bold 28px Arial';
  ctx.fillText(text, 60, 70);
}

function handleDemoClick(event) {
  const key = event.currentTarget.dataset.demo;
  const response = demoResponses[key];
  if (!response) return;
  handValue.textContent = response.gesture || 'Unknown';
  faceValue.textContent = response.expression || 'Unknown';
  gestureConfidenceEl.textContent = '92%';
  expressionConfidenceEl.textContent = '89%';
  intentValue.textContent = 'Demo';
  setMessage(response.text);
  drawDemoOverlay(response.text);
}

function attachDemoHandlers() {
  document.querySelectorAll('.demo-btn').forEach((btn) => {
    btn.addEventListener('click', handleDemoClick);
  });
}

function loadHistory() {
  updateHistory();
}

function registerControls() {
  document.getElementById('startCameraBtn').addEventListener('click', startCamera);
  document.getElementById('stopCameraBtn').addEventListener('click', stopCamera);
  document.getElementById('copyBtn').addEventListener('click', () => navigator.clipboard.writeText(messageOutput.value));
  document.getElementById('clearBtn').addEventListener('click', () => {
    state.composedMessage = '';
    state.lastEventKey = '';
    state.lastEmittedText = '';
    messageOutput.value = 'Gesture unclear. Please try again.';
  });
  document.getElementById('speakBtn').addEventListener('click', speakText);
  document.getElementById('addToMessageBtn').addEventListener('click', () => {
    const message = messageOutput.value.trim();
    if (message) addMessageToHistory(message);
  });
  document.getElementById('sendBtn').addEventListener('click', () => {
    const message = messageOutput.value.trim();
    if (message) {
      addMessageToHistory(message);
      alert('Message queued for sending.');
    }
  });
  document.getElementById('undoBtn').addEventListener('click', () => {
    state.recentMessages.pop();
    localStorage.setItem('gestureMessages', JSON.stringify(state.recentMessages));
    updateHistory();
  });
}

function animateFps() {
  const start = performance.now();
  requestAnimationFrame(function loop() {
    const now = performance.now();
    const fps = Math.min(30, Math.max(0, Math.round(1000 / (now - start || 1))));
    fpsValue.textContent = String(fps);
    requestAnimationFrame(loop);
  });
}

registerControls();
attachDemoHandlers();
loadHistory();
animateFps();

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  cameraStatus.textContent = 'Unsupported browser';
  state.demoMode = true;
}
