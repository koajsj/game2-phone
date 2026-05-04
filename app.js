import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22";

const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d", { alpha: true });

const statusText = document.getElementById("statusText");
const toggleBtn = document.getElementById("toggleBtn");
const scoreValue = document.getElementById("scoreValue");
const qualityTag = document.getElementById("qualityTag");
const headTilt = document.getElementById("headTilt");
const shoulderBalance = document.getElementById("shoulderBalance");
const trunkTilt = document.getElementById("trunkTilt");
const stability = document.getElementById("stability");
const sessionTime = document.getElementById("sessionTime");
const goodRatio = document.getElementById("goodRatio");
const alertTime = document.getElementById("alertTime");

const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

let poseLandmarker;
let stream;
let rafId;
let running = false;
let lastVideoTime = -1;
let lastFrameTs = 0;
let smoothScore = 0;
let smoothSpeed = 0;
let prevShoulderMid = null;

const session = {
  startAt: 0,
  goodMs: 0,
  alertMs: 0,
  totalMs: 0
};

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function mapPenalty(value, goodThreshold, maxThreshold) {
  if (value <= goodThreshold) return 0;
  return clamp((value - goodThreshold) / (maxThreshold - goodThreshold), 0, 1);
}

function calcDegAgainstVertical(from, to) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const cos = clamp((-dy) / len, -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

function formatMs(ms) {
  const sec = Math.max(0, Math.floor(ms / 1000));
  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = String(sec % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function resizeCanvasToDisplaySize() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function drawLandmarkPoint(p, color) {
  ctx.beginPath();
  ctx.arc(p.x * canvas.clientWidth, p.y * canvas.clientHeight, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawSegment(a, b, color) {
  ctx.beginPath();
  ctx.moveTo(a.x * canvas.clientWidth, a.y * canvas.clientHeight);
  ctx.lineTo(b.x * canvas.clientWidth, b.y * canvas.clientHeight);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function updateQualityTag(quality) {
  qualityTag.className = "tag";
  if (quality === "Good") {
    qualityTag.classList.add("good");
  } else if (quality === "Watch") {
    qualityTag.classList.add("warn");
  } else if (quality === "Alert") {
    qualityTag.classList.add("alert");
  } else {
    qualityTag.classList.add("neutral");
  }
  qualityTag.textContent = quality;
}

function getPoseMetrics(landmarks, deltaMs) {
  const nose = landmarks[0];
  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const lHip = landmarks[23];
  const rHip = landmarks[24];

  if (!nose || !lShoulder || !rShoulder || !lHip || !rHip) {
    return null;
  }

  const shoulderMid = { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 };
  const hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };

  const headDrift = calcDegAgainstVertical(shoulderMid, nose);
  const trunk = calcDegAgainstVertical(hipMid, shoulderMid);
  const shoulderGap = Math.abs(lShoulder.y - rShoulder.y) * 100;

  if (prevShoulderMid) {
    const speed = Math.hypot(shoulderMid.x - prevShoulderMid.x, shoulderMid.y - prevShoulderMid.y) /
      Math.max(deltaMs, 1);
    smoothSpeed = smoothSpeed * 0.8 + speed * 0.2;
  }
  prevShoulderMid = shoulderMid;

  const pHead = mapPenalty(headDrift, 15, 42);
  const pShoulder = mapPenalty(shoulderGap, 2.8, 10);
  const pTrunk = mapPenalty(trunk, 8, 24);
  const pStability = mapPenalty(smoothSpeed * 1300, 4, 20);
  const penalty = 100 * (0.36 * pHead + 0.24 * pShoulder + 0.28 * pTrunk + 0.12 * pStability);
  const rawScore = clamp(100 - penalty, 0, 100);
  smoothScore = smoothScore * 0.84 + rawScore * 0.16;

  const quality = smoothScore >= 82 ? "Good" : smoothScore >= 65 ? "Watch" : "Alert";

  return {
    score: smoothScore,
    quality,
    headDrift,
    shoulderGap,
    trunk,
    stability: clamp(100 - smoothSpeed * 2200, 0, 100),
    shoulderMid,
    hipMid,
    lShoulder,
    rShoulder,
    lHip,
    rHip,
    nose
  };
}

function renderPose(metrics) {
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  if (!metrics) return;

  const color =
    metrics.quality === "Good" ? "#3ddc97" : metrics.quality === "Watch" ? "#ffc857" : "#ff6978";

  drawSegment(metrics.lShoulder, metrics.rShoulder, color);
  drawSegment(metrics.lHip, metrics.rHip, color);
  drawSegment(metrics.shoulderMid, metrics.hipMid, color);
  drawSegment(metrics.shoulderMid, metrics.nose, color);

  drawLandmarkPoint(metrics.nose, "#6ce4ff");
  drawLandmarkPoint(metrics.lShoulder, color);
  drawLandmarkPoint(metrics.rShoulder, color);
  drawLandmarkPoint(metrics.lHip, color);
  drawLandmarkPoint(metrics.rHip, color);
}

function updateUI(metrics, deltaMs) {
  if (!metrics) {
    statusText.textContent = "No body detected. Move into frame.";
    return;
  }

  statusText.textContent =
    metrics.quality === "Good"
      ? "Posture stable."
      : metrics.quality === "Watch"
        ? "Minor drift detected."
        : "Adjust neck and shoulders.";

  scoreValue.textContent = String(Math.round(metrics.score));
  headTilt.textContent = `${metrics.headDrift.toFixed(1)}°`;
  shoulderBalance.textContent = `${metrics.shoulderGap.toFixed(1)}%`;
  trunkTilt.textContent = `${metrics.trunk.toFixed(1)}°`;
  stability.textContent = `${Math.round(metrics.stability)}%`;
  updateQualityTag(metrics.quality);

  session.totalMs += deltaMs;
  if (metrics.quality === "Good") session.goodMs += deltaMs;
  if (metrics.quality === "Alert") session.alertMs += deltaMs;

  sessionTime.textContent = formatMs(session.totalMs);
  goodRatio.textContent = `${Math.round((session.goodMs / Math.max(session.totalMs, 1)) * 100)}%`;
  alertTime.textContent = formatMs(session.alertMs);
}

async function initPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;
  statusText.textContent = "Loading pose model...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });
  return poseLandmarker;
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 60 }
    },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
}

function resetSessionState() {
  session.startAt = performance.now();
  session.totalMs = 0;
  session.goodMs = 0;
  session.alertMs = 0;
  smoothScore = 80;
  smoothSpeed = 0;
  prevShoulderMid = null;
  lastVideoTime = -1;
  lastFrameTs = performance.now();
}

function loop(ts) {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  resizeCanvasToDisplaySize();

  const deltaMs = Math.min(66, ts - lastFrameTs || 16.6);
  lastFrameTs = ts;

  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const result = poseLandmarker.detectForVideo(video, ts);
  const landmarks = result.landmarks?.[0];
  const metrics = landmarks ? getPoseMetrics(landmarks, deltaMs) : null;
  renderPose(metrics);
  updateUI(metrics, deltaMs);
}

async function toggleRun() {
  if (running) {
    running = false;
    toggleBtn.textContent = "Start";
    statusText.textContent = "Camera paused";
    cancelAnimationFrame(rafId);
    stopCamera();
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    return;
  }

  try {
    toggleBtn.disabled = true;
    toggleBtn.textContent = "Starting...";
    await initPoseLandmarker();
    await startCamera();
    resetSessionState();
    running = true;
    toggleBtn.textContent = "Stop";
    statusText.textContent = "Analyzing posture...";
    rafId = requestAnimationFrame(loop);
  } catch (err) {
    statusText.textContent = `Failed to start: ${err.message || err}`;
  } finally {
    toggleBtn.disabled = false;
    if (!running) toggleBtn.textContent = "Start";
  }
}

toggleBtn.addEventListener("click", toggleRun);
window.addEventListener("resize", resizeCanvasToDisplaySize);

if (!("mediaDevices" in navigator) || !("getUserMedia" in navigator.mediaDevices)) {
  statusText.textContent = "Your browser does not support camera APIs.";
  toggleBtn.disabled = true;
}
