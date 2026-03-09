/* ================================================================
   Palm Access — Biometric identification
   Browser-side: MediaPipe hand detection + client ROI crop
================================================================ */

import { HandLandmarker, FilesetResolver, DrawingUtils }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

// ── Timing constants ─────────────────────────────────────────────
const SCAN_HOLD_MS     = 800;    // hold steady before auto-scan triggers
const REG_HOLD_MS      = 1000;   // hold steady before auto-capture
const REG_COOLDOWN_MS  = 1500;   // gap between auto-captures
const SCAN_COOLDOWN_MS = 3000;   // cooldown after scan result

// Ring circumference for r=42: 2π×42 ≈ 263.9
const RING_C = 2 * Math.PI * 42;

// MediaPipe landmark indices (must match server)
const WRIST = 0, INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;

// ── State ────────────────────────────────────────────────────────
const state = {
  stream: null,
  currentTab: 'scan',
  autoMode: true,
  capturedImages: [],   // [{ data: base64, isRoi: bool }]
  handSeenMs: 0,
  lastFrameTs: null,
  lastLandmarks: null,  // most recent mediapipe landmarks for ROI extraction
  scanBusy: false,
  scanCooldownUntil: 0,
  regCooldownUntil: 0,
  scanStats: { total: 0, allowed: 0, denied: 0, users: 0 },
};

// ── DOM refs ─────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const video            = $('video');
const videoReg         = $('videoReg');
const canvas           = $('canvas');
const overlayCanvas    = $('overlayCanvas');
const overlayCanvasReg = $('overlayCanvasReg');
const btnScan          = $('btnScan');
const btnMode          = $('btnMode');
const btnCapture       = $('btnCapture');
const btnRegister      = $('btnRegister');
const btnRefresh       = $('btnRefresh');
const btnReset         = $('btnReset');
const userName         = $('userName');

// ── MediaPipe init ───────────────────────────────────────────────
let handLandmarker = null;
let drawUtils      = null;

async function initMediaPipe() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence: 0.4,
      minTrackingConfidence: 0.4,
    });

    drawUtils = new DrawingUtils(overlayCanvas.getContext('2d'));
    console.log('[PalmAccess] MediaPipe HandLandmarker ready');
    $('cameraStatus').innerHTML = '<span class="cam-dot"></span>Ready';
    startDetectLoop();
  } catch (err) {
    console.warn('[PalmAccess] MediaPipe failed — falling back to manual mode', err);
    setAutoMode(false);
    $('cameraStatus').textContent = 'Manual mode';
  }
}

// ── Detection loop ───────────────────────────────────────────────
function startDetectLoop() {
  requestAnimationFrame(detectLoop);
}

function detectLoop(ts) {
  requestAnimationFrame(detectLoop);
  if (!handLandmarker) return;

  const activeVideo = state.currentTab === 'scan' ? video : videoReg;
  if (activeVideo.readyState < 2 || activeVideo.paused) return;

  const result = handLandmarker.detectForVideo(activeVideo, ts);

  const activeCanvas = state.currentTab === 'scan' ? overlayCanvas : overlayCanvasReg;
  syncCanvasSize(activeCanvas, activeVideo);
  drawLandmarks(result, activeCanvas);

  const idleCanvas = state.currentTab === 'scan' ? overlayCanvasReg : overlayCanvas;
  idleCanvas.getContext('2d').clearRect(0, 0, idleCanvas.width, idleCanvas.height);

  const handFound = result.landmarks && result.landmarks.length > 0;
  const dt = state.lastFrameTs != null ? ts - state.lastFrameTs : 0;
  state.lastFrameTs = ts;

  if (handFound) {
    state.lastLandmarks = result.landmarks;
    state.handSeenMs = Math.min(state.handSeenMs + dt, Math.max(SCAN_HOLD_MS, REG_HOLD_MS) + 50);
    setCameraHandState(true);
    updateRingProgress();
    runAutoLogic();
    // Show brightness quality indicator
    if (state.currentTab === 'scan') {
      updateBrightnessBadge(video, result.landmarks[0], 'brightnessBadge');
    } else {
      updateBrightnessBadge(videoReg, result.landmarks[0], 'brightnessBadgeReg');
    }
  } else {
    state.handSeenMs = Math.max(0, state.handSeenMs - dt * 2.5);
    if (state.handSeenMs <= 0) {
      setCameraHandState(false);
      state.lastLandmarks = null;
      $('brightnessBadge').style.display    = 'none';
      $('brightnessBadgeReg').style.display = 'none';
    }
    updateRingProgress();
  }
}

function syncCanvasSize(cvs, vid) {
  if (cvs.width !== vid.videoWidth || cvs.height !== vid.videoHeight) {
    cvs.width  = vid.videoWidth  || 640;
    cvs.height = vid.videoHeight || 480;
  }
}

function drawLandmarks(result, cvs) {
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);

  if (!result.landmarks || !result.landmarks.length) return;

  const du = new DrawingUtils(ctx);
  for (const lm of result.landmarks) {
    du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, {
      color: '#6b7cf9',
      lineWidth: 2,
    });
    du.drawLandmarks(lm, {
      color: '#6b7cf9',
      fillColor: '#6b7cf9',
      lineWidth: 1,
      radius: 2,
    });
  }
}

// ── Brightness feedback ───────────────────────────────────────────
// Reads the mean luminance of the palm ROI canvas and updates a small
// badge so users know whether lighting conditions are suitable.
function updateBrightnessBadge(videoEl, landmarks, badgeId) {
  const badge = $(badgeId);
  if (!badge) return;
  if (!landmarks) { badge.style.display = 'none'; return; }

  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;
  const wrist     = landmarks[WRIST];
  const indexMcp  = landmarks[INDEX_MCP];
  const middleMcp = landmarks[MIDDLE_MCP];
  const pinkyMcp  = landmarks[PINKY_MCP];

  const cx = Math.round(middleMcp.x * w);
  const cy = Math.round(((middleMcp.y + wrist.y) / 2) * h);
  const palmWidth = Math.abs(Math.round((indexMcp.x - pinkyMcp.x) * w));
  const roiSize = Math.max(Math.round(palmWidth * 1.5), 60);
  const half = Math.round(roiSize / 2);
  const x1 = Math.max(0, cx - half);
  const y1 = Math.max(0, cy - half);
  const cropW = Math.min(w, cx + half) - x1;
  const cropH = Math.min(h, cy + half) - y1;
  if (cropW <= 0 || cropH <= 0) { badge.style.display = 'none'; return; }

  // Sample into a tiny 32×32 canvas to keep this cheap
  const tmp = document.createElement('canvas');
  tmp.width = 32; tmp.height = 32;
  tmp.getContext('2d').drawImage(videoEl, x1, y1, cropW, cropH, 0, 0, 32, 32);
  const pixels = tmp.getContext('2d').getImageData(0, 0, 32, 32).data;

  let sum = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    // Rec.709 luminance weights
    sum += pixels[i] * 0.2126 + pixels[i+1] * 0.7152 + pixels[i+2] * 0.0722;
  }
  const mean = sum / (pixels.length / 4);

  let label, cls;
  if (mean < 55) {
    label = 'Too dark'; cls = 'bri-dark';
  } else if (mean > 200) {
    label = 'Too bright'; cls = 'bri-bright';
  } else {
    label = 'Good light'; cls = 'bri-good';
  }

  badge.textContent = label;
  badge.className = `brightness-badge ${cls}`;
  badge.style.display = 'block';
}

// ── Client-side ROI extraction ───────────────────────────────────
// Mirrors the server's extract_palm_roi() logic, using landmarks already
// computed by the browser's MediaPipe instance. Sends a small JPEG crop
// instead of a full-resolution PNG, eliminating server-side detection.
function extractClientROI(videoEl, landmarks) {
  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;

  const wrist     = landmarks[WRIST];
  const indexMcp  = landmarks[INDEX_MCP];
  const middleMcp = landmarks[MIDDLE_MCP];
  const pinkyMcp  = landmarks[PINKY_MCP];

  const cx = Math.round(middleMcp.x * w);
  const cy = Math.round(((middleMcp.y + wrist.y) / 2) * h);
  const palmWidth = Math.abs(Math.round((indexMcp.x - pinkyMcp.x) * w));
  const roiSize = Math.max(Math.round(palmWidth * 1.5), 60);
  const half = Math.round(roiSize / 2);

  const x1 = Math.max(0, cx - half);
  const y1 = Math.max(0, cy - half);
  const x2 = Math.min(w, cx + half);
  const y2 = Math.min(h, cy + half);

  const roiCanvas = document.createElement('canvas');
  roiCanvas.width  = x2 - x1;
  roiCanvas.height = y2 - y1;
  roiCanvas.getContext('2d').drawImage(videoEl, x1, y1, x2 - x1, y2 - y1, 0, 0, x2 - x1, y2 - y1);

  return roiCanvas.toDataURL('image/jpeg', 0.9);
}

// ── Ring progress ────────────────────────────────────────────────
function updateRingProgress() {
  const tab = state.currentTab;

  if (tab === 'scan') {
    const holdMs = SCAN_HOLD_MS;
    const ring   = $('autoscanRing');
    const fill   = $('ringFill');
    const label  = $('ringLabel');
    const pct    = Math.min(state.handSeenMs / holdMs, 1);

    if (state.handSeenMs > 20) {
      ring.style.display = 'block';
      fill.style.strokeDashoffset = RING_C * (1 - pct);
      const remaining = Math.ceil((holdMs - state.handSeenMs) / 1000);
      label.textContent = pct >= 1 ? '✓' : (remaining > 0 ? remaining + 's' : 'Hold');
    } else {
      ring.style.display = 'none';
    }
  }

  if (tab === 'register') {
    const holdMs = REG_HOLD_MS;
    const ring   = $('autoscanRingReg');
    const fill   = $('ringFillReg');
    const label  = $('ringLabelReg');
    const pct    = Math.min(state.handSeenMs / holdMs, 1);

    if (state.handSeenMs > 20 && state.capturedImages.length < 5 && Date.now() >= state.regCooldownUntil) {
      ring.style.display = 'block';
      fill.style.strokeDashoffset = RING_C * (1 - pct);
      label.textContent = pct >= 1 ? '✓' : 'Hold';
    } else {
      ring.style.display = 'none';
    }
  }
}

// ── Auto-logic trigger ───────────────────────────────────────────
function runAutoLogic() {
  if (!state.autoMode) return;

  const now = Date.now();

  if (state.currentTab === 'scan') {
    if (state.handSeenMs >= SCAN_HOLD_MS && !state.scanBusy && now >= state.scanCooldownUntil) {
      triggerScan();
    }
  }

  if (state.currentTab === 'register') {
    const hasName = userName.value.trim().length > 0;
    if (
      state.handSeenMs >= REG_HOLD_MS &&
      state.capturedImages.length < 5 &&
      now >= state.regCooldownUntil &&
      hasName
    ) {
      triggerCapture();
      state.regCooldownUntil = now + REG_COOLDOWN_MS;
      state.handSeenMs = 0;
    }
  }
}

// ── Camera hand-detected visual state ────────────────────────────
function setCameraHandState(detected) {
  const frame = state.currentTab === 'scan' ? $('cameraFrame') : $('regCameraFrame');
  if (!frame) return;
  frame.classList.toggle('hand-detected', detected);

  if (state.currentTab === 'scan') {
    $('palmGuide').style.opacity = detected ? '0' : '1';
  } else {
    $('palmGuideReg').style.opacity = detected ? '0' : '1';
  }
}

// ── Webcam ───────────────────────────────────────────────────────
async function startCamera() {
  if (state.stream) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject    = state.stream;
    videoReg.srcObject = state.stream;
    $('cameraStatus').innerHTML = '<span class="cam-dot"></span>Loading detector…';
  } catch (err) {
    $('cameraStatus').textContent = 'Camera error';
    console.error('Camera error:', err);
  }
}

function captureFrame(videoEl) {
  const w = videoEl.videoWidth  || 640;
  const h = videoEl.videoHeight || 480;
  canvas.width  = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(videoEl, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.9);
}

function triggerFlash(flashId) {
  const el = $(flashId);
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

// ── Tab navigation ───────────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  state.currentTab = tab;
  state.handSeenMs = 0;

  document.querySelectorAll('.nav-btn').forEach((b) =>
    b.classList.toggle('active', b.dataset.tab === tab)
  );
  document.querySelectorAll('.panel').forEach((p) =>
    p.classList.toggle('active', p.id === `panel-${tab}`)
  );

  if (tab === 'log') {
    loadLogs();
    loadUsers();
  }
}

// ── Auto / Manual mode toggle ────────────────────────────────────
btnMode.addEventListener('click', () => setAutoMode(!state.autoMode));

function setAutoMode(on) {
  state.autoMode = on;
  btnMode.textContent = on ? 'Auto' : 'Manual';
  btnMode.classList.toggle('manual', !on);
  $('idleText').innerHTML = on
    ? 'Hold your open palm<br/>in front of the camera'
    : 'Press <strong>Scan now</strong><br/>to identify your palm';
  $('idleHint') && ($('idleHint').textContent = on ? 'Auto-detect on' : 'Manual mode');
}

// ── Scan Palm ────────────────────────────────────────────────────
btnScan.addEventListener('click', () => {
  if (!state.scanBusy) triggerScan();
});

async function triggerScan() {
  if (state.scanBusy) return;
  state.scanBusy = true;
  state.handSeenMs = 0;
  $('autoscanRing').style.display = 'none';

  triggerFlash('captureFlash');
  showScanning();

  // Prefer client-side ROI crop (skips server MediaPipe, saves ~500–1500ms)
  let b64, isRoi;
  if (state.lastLandmarks && state.lastLandmarks.length > 0) {
    b64   = extractClientROI(video, state.lastLandmarks[0]);
    isRoi = true;
  } else {
    b64   = captureFrame(video);
    isRoi = false;
  }

  const scanStart = performance.now();

  try {
    const res = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: b64, is_roi: isRoi }),
    });
    const elapsed = Math.round(performance.now() - scanStart);

    if (res.status === 422) {
      showNoHand('No hand detected — adjust position and try again');
    } else if (!res.ok) {
      showNoHand('Server error — please try again');
    } else {
      const data = await res.json();
      showResult(data, elapsed);
      updateStats(data.status);
    }
  } catch (err) {
    showNoHand('Network error');
    console.error(err);
  }

  state.scanCooldownUntil = Date.now() + SCAN_COOLDOWN_MS;
  state.scanBusy = false;
}

function showScanning() {
  $('resultDisplay').style.display  = 'none';
  $('resultIdle').style.display     = 'none';
  $('resultScanning').style.display = 'flex';
  $('resultCard').className = 'result-card';
}

function showNoHand(msg) {
  $('resultScanning').style.display = 'none';
  $('resultDisplay').style.display  = 'none';
  $('resultIdle').style.display     = 'flex';
  $('resultIdle').querySelector('.idle-text').innerHTML =
    msg + '<br/><small style="opacity:.6;font-size:.85em">Adjust and try again</small>';
  $('resultCard').className = 'result-card';
}

function showResult(data, elapsedMs) {
  $('resultScanning').style.display = 'none';
  $('resultIdle').style.display     = 'none';
  $('resultDisplay').style.display  = 'flex';

  const ok = data.status === 'ALLOWED';
  $('resultCard').className = `result-card ${ok ? 'allowed' : 'denied'}`;
  $('badgeIcon').textContent   = ok ? '✓' : '✕';
  $('badgeStatus').textContent = ok ? 'Allowed' : 'Denied';
  $('badgeStatus').className   = `badge-status ${ok ? 'allowed' : 'denied'}`;
  $('resultName').textContent  = ok ? data.name : 'Unrecognized';
  $('resultName').className    = `result-name ${ok ? 'allowed' : 'denied'}`;
  $('resultSimilarity').textContent =
    data.similarity != null ? (data.similarity * 100).toFixed(1) + '%' : '—';

  const closestRow = $('closestRow');
  if (!ok && data.closest_match) {
    closestRow.style.display = 'flex';
    $('resultClosest').textContent =
      data.closest_match + ' (' + (data.similarity * 100).toFixed(1) + '%)';
  } else {
    closestRow.style.display = 'none';
  }

  $('resultTimestamp').textContent = new Date().toLocaleTimeString();

  const timingRow = $('timingRow');
  if (elapsedMs != null) {
    $('resultTiming').textContent = elapsedMs + ' ms';
    timingRow.style.display = 'flex';
  } else {
    timingRow.style.display = 'none';
  }
}

function updateStats(status) {
  state.scanStats.total++;
  if (status === 'ALLOWED') state.scanStats.allowed++;
  else state.scanStats.denied++;
  $('statTotal').textContent   = state.scanStats.total;
  $('statAllowed').textContent = state.scanStats.allowed;
  $('statDenied').textContent  = state.scanStats.denied;
}

// ── Register ─────────────────────────────────────────────────────
btnCapture.addEventListener('click', () => triggerCapture());

btnReset?.addEventListener('click', () => {
  const savedName = userName.value;
  resetRegistration();
  userName.value = savedName;
  setFeedback('', '');
});

function triggerCapture() {
  if (state.capturedImages.length >= 5) return;

  let b64, isRoi;
  if (state.lastLandmarks && state.lastLandmarks.length > 0) {
    b64   = extractClientROI(videoReg, state.lastLandmarks[0]);
    isRoi = true;
  } else {
    b64   = captureFrame(videoReg);
    isRoi = false;
  }

  triggerFlash('captureFlashReg');
  state.capturedImages.push({ data: b64, isRoi });

  const count = state.capturedImages.length;
  $('captureCounter').textContent = `${count} / 5`;
  if (btnReset) btnReset.disabled = false;

  document.querySelectorAll('.dot').forEach((dot, i) =>
    dot.classList.toggle('filled', i < count)
  );

  if (count >= 5) {
    btnRegister.disabled = !userName.value.trim();
    $('registerHint').textContent = 'All 5 samples captured. Press Register.';
    $('autoscanRingReg').style.display = 'none';
  } else {
    $('registerHint').textContent =
      `${5 - count} more sample${5 - count > 1 ? 's' : ''} needed — hold still.`;
  }
}

userName.addEventListener('input', () => {
  btnRegister.disabled = state.capturedImages.length < 5 || !userName.value.trim();
});

btnRegister.addEventListener('click', async () => {
  const name = userName.value.trim();
  if (!name || state.capturedImages.length < 5) return;

  btnRegister.disabled = true;
  setFeedback('Registering…', '');

  const allRoi = state.capturedImages.every((c) => c.isRoi);

  try {
    const res = await fetch('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        images: state.capturedImages.map((c) => c.data),
        is_roi: allRoi,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      const msg = data.detail || 'Registration failed';
      setFeedback(msg + ' — samples cleared, please try again.', 'error');
      // Auto-reset after a short pause so user can read the error
      setTimeout(() => {
        const savedName = userName.value;
        resetRegistration();
        userName.value = savedName;   // keep the name they typed
        setFeedback('Ready — recapture all 5 samples.', '');
      }, 2500);
      return;
    }

    setFeedback(`✓ ${data.name} enrolled successfully`, 'success');
    resetRegistration();
    loadStats();
  } catch (err) {
    setFeedback('Network error — samples cleared, please try again.', 'error');
    setTimeout(() => {
      const savedName = userName.value;
      resetRegistration();
      userName.value = savedName;
      setFeedback('', '');
    }, 2500);
    console.error(err);
  }
});

function resetRegistration() {
  state.capturedImages = [];
  state.handSeenMs = 0;
  userName.value = '';
  $('captureCounter').textContent = '0 / 5';
  document.querySelectorAll('.dot').forEach((d) => d.classList.remove('filled'));
  $('registerHint').textContent = 'Enter a name, then hold your palm in front of the camera.';
  btnRegister.disabled = true;
  if (btnReset) btnReset.disabled = true;
  $('autoscanRingReg').style.display = 'none';
}

function setFeedback(msg, type) {
  const el = $('registerFeedback');
  el.textContent = msg;
  el.className = `register-feedback ${type}`;
}

// ── Access Log ───────────────────────────────────────────────────
btnRefresh.addEventListener('click', () => { loadLogs(); loadUsers(); });

async function loadLogs() {
  try {
    const logs = await fetch('/api/logs?limit=100').then((r) => r.json());
    renderLogs(logs);
  } catch (err) { console.error(err); }
}

function renderLogs(logs) {
  const tbody = $('logTableBody');
  if (!logs.length) {
    tbody.innerHTML = `<tr class="log-empty-row"><td colspan="4"><div class="log-empty">
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
        <path d="M8 12h24M8 20h16M8 28h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg><span>No access attempts recorded yet</span></div></td></tr>`;
    return;
  }
  tbody.innerHTML = logs.map((log) => {
    const ts = new Date(log.timestamp);
    const ok = log.status === 'ALLOWED';
    return `<tr>
      <td>${isNaN(ts) ? log.timestamp : ts.toLocaleString()}</td>
      <td>${esc(log.matched_name)}</td>
      <td><span class="log-status ${ok ? 'allowed' : 'denied'}">${ok ? 'Allowed' : 'Denied'}</span></td>
      <td>${log.similarity != null ? (log.similarity * 100).toFixed(1) + '%' : '—'}</td>
    </tr>`;
  }).join('');
}

async function loadUsers() {
  try {
    const users = await fetch('/api/users').then((r) => r.json());
    renderUsers(users);
    state.scanStats.users = users.length;
    $('statUsers').textContent = users.length;
  } catch (err) { console.error(err); }
}

function renderUsers(users) {
  const grid = $('usersGrid');
  if (!users.length) {
    grid.innerHTML = '<div class="users-empty">No users enrolled yet.</div>';
    return;
  }
  grid.innerHTML = users.map((u) => `
    <div class="user-chip" id="chip-${u.id}">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <circle cx="7" cy="4.5" r="2.5" stroke="currentColor" stroke-width="1.2"/>
        <path d="M2 13c0-2.761 2.239-4 5-4s5 1.239 5 4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
      <span class="user-chip-name">${esc(u.name)}</span>
      <button class="user-chip-delete" onclick="window.deleteUser(${u.id})" title="Remove">×</button>
    </div>`).join('');
}

window.deleteUser = async (id) => {
  if (!confirm('Remove this user?')) return;
  await fetch(`/api/users/${id}`, { method: 'DELETE' });
  loadUsers();
  loadLogs();
};

async function loadStats() {
  try {
    const users = await fetch('/api/users').then((r) => r.json());
    $('statUsers').textContent = users.length;
  } catch (_) { /* silent */ }
}

const esc = (s) =>
  String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ── Init ─────────────────────────────────────────────────────────
(async () => {
  await startCamera();
  loadStats();
  video.addEventListener('loadeddata', () => initMediaPipe(), { once: true });
  if (video.readyState >= 2) initMediaPipe();
})();
