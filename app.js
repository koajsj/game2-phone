const ui = {
  video: document.getElementById("camera"),
  overlay: document.getElementById("overlay"),
  statusText: document.getElementById("statusText"),
  modePill: document.getElementById("modePill"),
  toggleBtn: document.getElementById("toggleBtn"),
  resetBtn: document.getElementById("resetBtn"),
  cameraEmpty: document.getElementById("cameraEmpty"),
  scoreCard: document.getElementById("scoreCard"),
  scoreRing: document.getElementById("scoreRing"),
  scoreValue: document.getElementById("scoreValue"),
  qualityTag: document.getElementById("qualityTag"),
  coachTitle: document.getElementById("coachTitle"),
  coachSummary: document.getElementById("coachSummary"),
  trendCaption: document.getElementById("trendCaption"),
  trendCanvas: document.getElementById("trendCanvas"),
  focusArea: document.getElementById("focusArea"),
  headTilt: document.getElementById("headTilt"),
  shoulderBalance: document.getElementById("shoulderBalance"),
  trunkTilt: document.getElementById("trunkTilt"),
  stability: document.getElementById("stability"),
  headTiltBar: document.getElementById("headTiltBar"),
  shoulderBalanceBar: document.getElementById("shoulderBalanceBar"),
  trunkTiltBar: document.getElementById("trunkTiltBar"),
  stabilityBar: document.getElementById("stabilityBar"),
  sessionState: document.getElementById("sessionState"),
  sessionTime: document.getElementById("sessionTime"),
  goodRatio: document.getElementById("goodRatio"),
  alertTime: document.getElementById("alertTime"),
  samplesSeen: document.getElementById("samplesSeen"),
  cueTitles: [
    document.getElementById("cue1Title"),
    document.getElementById("cue2Title"),
    document.getElementById("cue3Title")
  ],
  cueTexts: [
    document.getElementById("cue1Text"),
    document.getElementById("cue2Text"),
    document.getElementById("cue3Text")
  ]
};

const ctx = ui.overlay.getContext("2d", { alpha: true });
const trendCtx = ui.trendCanvas.getContext("2d");

const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const TASKS_VISION_MODULE_CANDIDATES = [
  {
    module: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs",
    wasm: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
  },
  {
    module: "https://unpkg.com/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs",
    wasm: "https://unpkg.com/@mediapipe/tasks-vision@0.10.35/wasm"
  }
];

const EMPTY_RECOMMENDATIONS = {
  focusArea: "等待画面",
  items: [
    {
      title: "先完整进入画面",
      text: "确保头部、肩部和髋部位于画面内，避免遮挡和侧身过大。"
    },
    {
      title: "保持设备稳定",
      text: "摄像头最好固定在屏幕上方，避免手持导致整体晃动。"
    },
    {
      title: "留出上半身空间",
      text: "让肩膀和胸廓都可见，系统才能稳定评估躯干与肩线。"
    }
  ]
};

const QUALITY_META = {
  waiting: {
    label: "待机",
    tagClass: "neutral",
    pillClass: "idle",
    title: "保持自然坐姿",
    summary: "检测启动后会持续计算姿态分数，并根据变化趋势刷新建议。"
  },
  good: {
    label: "优秀",
    tagClass: "good",
    pillClass: "running",
    title: "状态稳定",
    summary: "头部、肩线和躯干都保持在较好的范围内，继续维持即可。"
  },
  warn: {
    label: "注意",
    tagClass: "warn",
    pillClass: "warning",
    title: "开始偏移",
    summary: "当前姿态仍可恢复，优先处理最明显的偏移点，避免继续累积。"
  },
  alert: {
    label: "调整",
    tagClass: "alert",
    pillClass: "alert",
    title: "需要尽快修正",
    summary: "已经出现明显偏移，建议先回正头部和肩线，再恢复稳定呼吸。"
  },
  paused: {
    label: "暂停",
    tagClass: "neutral",
    pillClass: "idle",
    title: "已暂停分析",
    summary: "可以继续当前会话，或者直接重置后重新开始检测。"
  }
};

const state = {
  poseLandmarker: null,
  filesetResolver: null,
  poseLandmarkerClass: null,
  wasmRoot: "",
  stream: null,
  rafId: 0,
  running: false,
  lastVideoTime: -1,
  lastFrameTs: 0,
  smoothScore: 80,
  smoothSpeed: 0,
  prevShoulderMid: null,
  history: []
};

const session = {
  totalMs: 0,
  goodMs: 0,
  alertMs: 0,
  samplesSeen: 0
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mapPenalty(value, goodThreshold, maxThreshold) {
  if (value <= goodThreshold) return 0;
  return clamp((value - goodThreshold) / (maxThreshold - goodThreshold), 0, 1);
}

function calcDegAgainstVertical(from, to) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const cos = clamp(-dy / len, -1, 1);
  return (Math.acos(cos) * 180) / Math.PI;
}

function formatMs(ms) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = String(Math.floor(seconds / 60)).padStart(2, "0");
  const secs = String(seconds % 60).padStart(2, "0");
  return `${minutes}:${secs}`;
}

function setBodyState(mode) {
  document.body.dataset.state = mode;
}

function resizeOverlayToDisplaySize() {
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = Math.floor(ui.overlay.clientWidth * dpr);
  const displayHeight = Math.floor(ui.overlay.clientHeight * dpr);

  if (ui.overlay.width !== displayWidth || ui.overlay.height !== displayHeight) {
    ui.overlay.width = displayWidth;
    ui.overlay.height = displayHeight;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function clearOverlay() {
  ctx.clearRect(0, 0, ui.overlay.clientWidth, ui.overlay.clientHeight);
}

function resizeTrendCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const displayWidth = Math.floor(ui.trendCanvas.clientWidth * dpr);
  const displayHeight = Math.floor(ui.trendCanvas.clientHeight * dpr);

  if (!displayWidth || !displayHeight) return;

  if (ui.trendCanvas.width !== displayWidth || ui.trendCanvas.height !== displayHeight) {
    ui.trendCanvas.width = displayWidth;
    ui.trendCanvas.height = displayHeight;
    trendCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

function drawLandmarkPoint(point, color) {
  ctx.beginPath();
  ctx.arc(point.x * ui.overlay.clientWidth, point.y * ui.overlay.clientHeight, 4.6, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.fill();
  ctx.shadowBlur = 0;
}

function drawSegment(a, b, color) {
  ctx.beginPath();
  ctx.moveTo(a.x * ui.overlay.clientWidth, a.y * ui.overlay.clientHeight);
  ctx.lineTo(b.x * ui.overlay.clientWidth, b.y * ui.overlay.clientHeight);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.3;
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function setModePill(text, className) {
  ui.modePill.className = "mode-pill";
  ui.modePill.classList.add(className);
  ui.modePill.textContent = text;
}

function setQualityState(level) {
  const meta = QUALITY_META[level] || QUALITY_META.waiting;
  ui.qualityTag.className = "tag";
  ui.qualityTag.classList.add(meta.tagClass);
  ui.qualityTag.textContent = meta.label;
  ui.scoreCard.dataset.quality = meta.tagClass;
  ui.coachTitle.textContent = meta.title;
  ui.coachSummary.textContent = meta.summary;
  setModePill(meta.label, meta.pillClass);
}

function setScore(score) {
  if (score == null) {
    ui.scoreValue.textContent = "--";
    ui.scoreRing.style.setProperty("--score-angle", "0deg");
    return;
  }

  const safeScore = clamp(score, 0, 100);
  ui.scoreValue.textContent = String(Math.round(safeScore));
  ui.scoreRing.style.setProperty("--score-angle", `${safeScore * 3.6}deg`);
}

function setMetricValue(element, value, suffix) {
  element.textContent = value == null ? "--" : `${value}${suffix}`;
}

function setMetricBar(element, percent, mode = "good") {
  const safePercent = clamp(percent, 0, 100);
  element.style.width = `${safePercent}%`;
  element.style.background =
    mode === "alert"
      ? "linear-gradient(90deg, #ff7c70, #ffb29f)"
      : mode === "warn"
        ? "linear-gradient(90deg, #ffbf5f, #ffe29d)"
        : "linear-gradient(90deg, #75dff2, #94f3d0)";
}

function updateTrendCaption() {
  if (state.history.length < 4) {
    ui.trendCaption.textContent = "等待数据";
    return;
  }

  const recent = state.history.slice(-6);
  const diff = recent[recent.length - 1] - recent[0];
  if (diff > 4) ui.trendCaption.textContent = "趋势回升";
  else if (diff < -4) ui.trendCaption.textContent = "趋势下滑";
  else ui.trendCaption.textContent = "趋势平稳";
}

function drawTrend() {
  resizeTrendCanvas();

  const width = ui.trendCanvas.clientWidth;
  const height = ui.trendCanvas.clientHeight;
  trendCtx.clearRect(0, 0, width, height);

  trendCtx.fillStyle = "rgba(255, 255, 255, 0.04)";
  trendCtx.fillRect(0, height - 1, width, 1);

  if (!state.history.length) {
    trendCtx.fillStyle = "rgba(151, 182, 194, 0.9)";
    trendCtx.font = "14px 'Noto Sans SC'";
    trendCtx.fillText("尚无趋势数据", 12, 52);
    return;
  }

  const values = state.history.slice(-36);
  const maxPoints = Math.max(values.length - 1, 1);
  const stepX = (width - 24) / maxPoints;

  trendCtx.beginPath();
  values.forEach((value, index) => {
    const x = 12 + index * stepX;
    const y = 10 + ((100 - value) / 100) * (height - 20);
    if (index === 0) trendCtx.moveTo(x, y);
    else trendCtx.lineTo(x, y);
  });
  trendCtx.strokeStyle = "rgba(117, 223, 242, 0.95)";
  trendCtx.lineWidth = 3;
  trendCtx.lineJoin = "round";
  trendCtx.lineCap = "round";
  trendCtx.shadowColor = "rgba(117, 223, 242, 0.5)";
  trendCtx.shadowBlur = 12;
  trendCtx.stroke();
  trendCtx.shadowBlur = 0;

  const lastValue = values[values.length - 1];
  const lastX = 12 + (values.length - 1) * stepX;
  const lastY = 10 + ((100 - lastValue) / 100) * (height - 20);

  trendCtx.beginPath();
  trendCtx.arc(lastX, lastY, 4.6, 0, Math.PI * 2);
  trendCtx.fillStyle = "rgba(148, 243, 208, 1)";
  trendCtx.fill();
}

function updateCueCards(items) {
  items.slice(0, 3).forEach((item, index) => {
    ui.cueTitles[index].textContent = item.title;
    ui.cueTexts[index].textContent = item.text;
  });
}

function getRecommendations(metrics) {
  if (!metrics) return EMPTY_RECOMMENDATIONS;

  const severities = [
    { key: "head", value: metrics.headDriftDeg },
    { key: "shoulder", value: metrics.shoulderGapPct },
    { key: "trunk", value: metrics.trunkTiltDeg }
  ].sort((a, b) => b.value - a.value);

  const focusMap = {
    head: {
      focus: "优先修正头部前探",
      title: "收回下巴，放松后颈",
      text: "让耳朵回到肩膀正上方，避免头部持续向前探。"
    },
    shoulder: {
      focus: "优先拉平肩线",
      title: "让双肩重新对齐",
      text: "放松耸肩侧，轻收肩胛，避免一侧肩膀明显高于另一侧。"
    },
    trunk: {
      focus: "优先回正躯干",
      title: "让胸口与骨盆回到中线",
      text: "减少身体向一侧倾倒，保持胸廓和骨盆大致垂直。"
    }
  };

  const primary = focusMap[severities[0].key];
  const stabilityTip =
    metrics.stabilityPct < 70
      ? {
          title: "降低身体晃动",
          text: "双脚更稳地落地，前臂轻放桌面，减少频繁挪动和探身。"
        }
      : {
          title: "保持当前节奏",
          text: "已经接近稳定区间，保持自然呼吸，不要刻意僵硬挺直。"
        };

  const secondary =
    severities[1].key === "head"
      ? {
          title: "检查屏幕高度",
          text: "如果需要持续低头，抬高屏幕中心位置通常比强行抬头更有效。"
        }
      : severities[1].key === "shoulder"
        ? {
            title: "均衡左右受力",
            text: "避免长期把身体重心压在一侧手肘或一侧臀部。"
          }
        : {
            title: "收紧核心但不要憋气",
            text: "轻度激活腹部能帮助躯干回到中立位，不必大幅挺胸。"
          };

  return {
    focusArea: primary.focus,
    items: [primary, stabilityTip, secondary]
  };
}

function getPoseMetrics(landmarks, deltaMs) {
  const nose = landmarks[0];
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];

  if (!nose || !leftShoulder || !rightShoulder || !leftHip || !rightHip) {
    return null;
  }

  const shoulderMid = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2
  };
  const hipMid = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2
  };

  const headDriftDeg = calcDegAgainstVertical(shoulderMid, nose);
  const trunkTiltDeg = calcDegAgainstVertical(hipMid, shoulderMid);
  const shoulderGapPct = Math.abs(leftShoulder.y - rightShoulder.y) * 100;

  if (state.prevShoulderMid) {
    const speed =
      Math.hypot(
        shoulderMid.x - state.prevShoulderMid.x,
        shoulderMid.y - state.prevShoulderMid.y
      ) / Math.max(deltaMs, 1);
    state.smoothSpeed = state.smoothSpeed * 0.8 + speed * 0.2;
  }
  state.prevShoulderMid = shoulderMid;

  const headPenalty = mapPenalty(headDriftDeg, 15, 42);
  const shoulderPenalty = mapPenalty(shoulderGapPct, 2.8, 10);
  const trunkPenalty = mapPenalty(trunkTiltDeg, 8, 24);
  const stabilityPenalty = mapPenalty(state.smoothSpeed * 1300, 4, 20);

  const totalPenalty =
    100 *
    (0.36 * headPenalty +
      0.24 * shoulderPenalty +
      0.28 * trunkPenalty +
      0.12 * stabilityPenalty);

  const rawScore = clamp(100 - totalPenalty, 0, 100);
  state.smoothScore = state.smoothScore * 0.84 + rawScore * 0.16;

  let quality = "good";
  if (state.smoothScore < 65) quality = "alert";
  else if (state.smoothScore < 82) quality = "warn";

  return {
    score: state.smoothScore,
    quality,
    headDriftDeg,
    shoulderGapPct,
    trunkTiltDeg,
    stabilityPct: clamp(100 - state.smoothSpeed * 2200, 0, 100),
    shoulderMid,
    hipMid,
    leftShoulder,
    rightShoulder,
    leftHip,
    rightHip,
    nose
  };
}

function renderPose(metrics) {
  clearOverlay();
  if (!metrics) return;

  const color =
    metrics.quality === "good"
      ? "#4dd7a0"
      : metrics.quality === "warn"
        ? "#ffbf5f"
        : "#ff7c70";

  drawSegment(metrics.leftShoulder, metrics.rightShoulder, color);
  drawSegment(metrics.leftHip, metrics.rightHip, color);
  drawSegment(metrics.shoulderMid, metrics.hipMid, color);
  drawSegment(metrics.shoulderMid, metrics.nose, color);
  drawLandmarkPoint(metrics.nose, "#75dff2");
  drawLandmarkPoint(metrics.leftShoulder, color);
  drawLandmarkPoint(metrics.rightShoulder, color);
  drawLandmarkPoint(metrics.leftHip, color);
  drawLandmarkPoint(metrics.rightHip, color);
}

function updateSession(metrics, deltaMs) {
  session.totalMs += deltaMs;
  if (metrics.quality === "good") session.goodMs += deltaMs;
  if (metrics.quality === "alert") session.alertMs += deltaMs;
  session.samplesSeen += 1;

  ui.sessionTime.textContent = formatMs(session.totalMs);
  ui.goodRatio.textContent = `${Math.round((session.goodMs / Math.max(session.totalMs, 1)) * 100)}%`;
  ui.alertTime.textContent = formatMs(session.alertMs);
  ui.samplesSeen.textContent = String(session.samplesSeen);
  ui.sessionState.textContent = session.totalMs > 0 ? "分析进行中" : "未开始";
}

function updateMetricsPanel(metrics) {
  if (!metrics) {
    setMetricValue(ui.headTilt, null, "");
    setMetricValue(ui.shoulderBalance, null, "");
    setMetricValue(ui.trunkTilt, null, "");
    setMetricValue(ui.stability, null, "");
    [ui.headTiltBar, ui.shoulderBalanceBar, ui.trunkTiltBar, ui.stabilityBar].forEach((bar) =>
      setMetricBar(bar, 0, "good")
    );
    return;
  }

  setMetricValue(ui.headTilt, metrics.headDriftDeg.toFixed(1), "°");
  setMetricValue(ui.shoulderBalance, metrics.shoulderGapPct.toFixed(1), "%");
  setMetricValue(ui.trunkTilt, metrics.trunkTiltDeg.toFixed(1), "°");
  setMetricValue(ui.stability, Math.round(metrics.stabilityPct), "%");

  setMetricBar(ui.headTiltBar, clamp((metrics.headDriftDeg / 42) * 100, 0, 100), metrics.quality);
  setMetricBar(
    ui.shoulderBalanceBar,
    clamp((metrics.shoulderGapPct / 10) * 100, 0, 100),
    metrics.quality
  );
  setMetricBar(ui.trunkTiltBar, clamp((metrics.trunkTiltDeg / 24) * 100, 0, 100), metrics.quality);
  setMetricBar(ui.stabilityBar, metrics.stabilityPct, metrics.quality === "alert" ? "alert" : "good");
}

function updateInsights(metrics, deltaMs) {
  if (!metrics) {
    ui.statusText.textContent = "未检测到完整上半身，请回到取景框中央。";
    ui.focusArea.textContent = EMPTY_RECOMMENDATIONS.focusArea;
    setQualityState("waiting");
    updateMetricsPanel(null);
    updateCueCards(EMPTY_RECOMMENDATIONS.items);
    return;
  }

  const qualityStatus =
    metrics.quality === "good"
      ? "姿态稳定，继续保持。"
      : metrics.quality === "warn"
        ? "出现轻微偏移，建议立即回正。"
        : "偏移明显，先调整头部和肩线。";

  ui.statusText.textContent = qualityStatus;
  setQualityState(metrics.quality);
  setScore(metrics.score);
  updateMetricsPanel(metrics);
  updateSession(metrics, deltaMs);

  state.history.push(metrics.score);
  state.history = state.history.slice(-36);
  updateTrendCaption();
  drawTrend();

  const recommendations = getRecommendations(metrics);
  ui.focusArea.textContent = recommendations.focusArea;
  updateCueCards(recommendations.items);
}

async function loadTasksVisionModule() {
  if (state.filesetResolver && state.poseLandmarkerClass) return;

  let lastError = null;

  for (const candidate of TASKS_VISION_MODULE_CANDIDATES) {
    try {
      ui.statusText.textContent = `正在加载识别引擎（${new URL(candidate.module).host}）...`;
      const mod = await import(candidate.module);
      state.filesetResolver = mod.FilesetResolver;
      state.poseLandmarkerClass = mod.PoseLandmarker;
      state.wasmRoot = candidate.wasm;
      return;
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(`识别引擎加载失败：${lastError?.message || lastError || "未知错误"}`);
}

async function initPoseLandmarker() {
  if (state.poseLandmarker) return state.poseLandmarker;

  await loadTasksVisionModule();
  ui.statusText.textContent = "正在加载姿态模型...";

  const vision = await state.filesetResolver.forVisionTasks(state.wasmRoot);

  try {
    state.poseLandmarker = await state.poseLandmarkerClass.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1
    });
  } catch {
    state.poseLandmarker = await state.poseLandmarkerClass.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1
    });
  }

  return state.poseLandmarker;
}

async function startCamera() {
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 60 }
    },
    audio: false
  });
  ui.video.srcObject = state.stream;
  await ui.video.play();
}

function stopCamera() {
  if (!state.stream) return;
  state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.video.srcObject = null;
}

function resetSessionState() {
  session.totalMs = 0;
  session.goodMs = 0;
  session.alertMs = 0;
  session.samplesSeen = 0;
  state.smoothScore = 80;
  state.smoothSpeed = 0;
  state.prevShoulderMid = null;
  state.lastVideoTime = -1;
  state.lastFrameTs = performance.now();
  state.history = [];
}

function resetDashboard() {
  setScore(null);
  setQualityState("waiting");
  ui.statusText.textContent = "点击开始，进行本地姿势检测。";
  ui.focusArea.textContent = "等待分析";
  ui.sessionState.textContent = "未开始";
  ui.sessionTime.textContent = "00:00";
  ui.goodRatio.textContent = "0%";
  ui.alertTime.textContent = "00:00";
  ui.samplesSeen.textContent = "0";
  updateMetricsPanel(null);
  updateCueCards(EMPTY_RECOMMENDATIONS.items);
  state.history = [];
  ui.trendCaption.textContent = "等待数据";
  drawTrend();
}

function setUiRunning(running) {
  state.running = running;
  ui.cameraEmpty.classList.toggle("is-hidden", running);
  ui.toggleBtn.textContent = running ? "停止检测" : "开始检测";
  ui.resetBtn.disabled = !running && session.totalMs === 0;
  setBodyState(running ? "running" : "idle");
}

function loop(ts) {
  if (!state.running) return;

  state.rafId = requestAnimationFrame(loop);
  resizeOverlayToDisplaySize();

  const deltaMs = Math.min(66, ts - state.lastFrameTs || 16.6);
  state.lastFrameTs = ts;

  if (ui.video.currentTime === state.lastVideoTime) return;
  state.lastVideoTime = ui.video.currentTime;

  const result = state.poseLandmarker.detectForVideo(ui.video, ts);
  const landmarks = result.landmarks?.[0];
  const metrics = landmarks ? getPoseMetrics(landmarks, deltaMs) : null;

  renderPose(metrics);
  updateInsights(metrics, deltaMs);
}

async function stopRun({ preserveMessage = true } = {}) {
  state.running = false;
  cancelAnimationFrame(state.rafId);
  stopCamera();
  clearOverlay();
  setUiRunning(false);
  if (preserveMessage) {
    ui.statusText.textContent = session.totalMs > 0 ? "检测已暂停，可继续或重置会话。" : "点击开始，进行本地姿势检测。";
    setQualityState(session.totalMs > 0 ? "paused" : "waiting");
  }
}

async function toggleRun() {
  if (state.running) {
    await stopRun();
    return;
  }

  try {
    ui.toggleBtn.disabled = true;
    ui.resetBtn.disabled = true;
    ui.toggleBtn.textContent = "正在启动...";
    ui.statusText.textContent = "正在准备摄像头和姿态模型...";
    await initPoseLandmarker();
    await startCamera();
    resetSessionState();
    resetDashboard();
    setUiRunning(true);
    ui.resetBtn.disabled = false;
    setQualityState("waiting");
    ui.statusText.textContent = "识别已启动，正在等待稳定画面...";
    state.rafId = requestAnimationFrame(loop);
  } catch (error) {
    await stopRun({ preserveMessage: false });
    ui.statusText.textContent = `启动失败：${error?.message || error}。请检查摄像头权限和网络访问。`;
    setQualityState("paused");
  } finally {
    ui.toggleBtn.disabled = false;
    ui.toggleBtn.textContent = state.running ? "停止检测" : "开始检测";
    ui.resetBtn.disabled = !state.running && session.totalMs === 0;
  }
}

function handleReset() {
  resetSessionState();
  resetDashboard();
  ui.resetBtn.disabled = !state.running;
  if (state.running) {
    ui.statusText.textContent = "会话已重置，继续分析中...";
    setQualityState("waiting");
  }
}

function handleVisibilityChange() {
  if (document.hidden && state.running) {
    stopRun();
  }
}

ui.toggleBtn.addEventListener("click", toggleRun);
ui.resetBtn.addEventListener("click", handleReset);
window.addEventListener("resize", () => {
  resizeOverlayToDisplaySize();
  drawTrend();
});
window.visualViewport?.addEventListener("resize", () => {
  resizeOverlayToDisplaySize();
  drawTrend();
});
window.addEventListener("beforeunload", () => {
  cancelAnimationFrame(state.rafId);
  stopCamera();
});
document.addEventListener("visibilitychange", handleVisibilityChange);

drawTrend();

if (!("mediaDevices" in navigator) || !("getUserMedia" in navigator.mediaDevices)) {
  ui.statusText.textContent = "当前浏览器不支持摄像头接口。";
  ui.toggleBtn.disabled = true;
}

if (!window.isSecureContext) {
  ui.statusText.textContent = "摄像头调用需要 HTTPS 或 localhost 环境。";
  ui.toggleBtn.disabled = true;
}
