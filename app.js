import {
  FilesetResolver,
  HandLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const DYNAMIC_MODEL_URL = "./lsm_dynamic_gru.onnx";
const BASELINE_URL = "./lsm_landmarks.json";

const DEFAULT_CLASSES = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I",
  "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y"
];
const DYNAMIC_LABELS = ["J", "K", "Ñ", "Q", "X", "Z", "STATIC_IDLE"];

const STORAGE_KEY = "lsm_knn_samples";
const LEGACY_STORAGE_KEY = "lsm_knn_samples_v1";

const NUM_LANDMARKS = 21;
const FEATURE_DIM = NUM_LANDMARKS * 3;
const FEATURE_VERSION = 2;
const STATIC_SPACE = "image";
const SCALE_REF_IDX = [9];
const FEATURE_TAG = `v${FEATURE_VERSION}:${STATIC_SPACE}:${SCALE_REF_IDX.join(",")}`;

const LEGACY_CAPTURE_SIZE = { width: 640, height: 480 };

//Static classifier (KNN)
const KNN_K = 5;
const KNN_EPS = 1e-4;
const MAX_MATCH_DIST = 1;

//Dynamic classifier (GRU)
const DYNAMIC_FEATURES = "legacy";
const SEQ_LEN = 30;
const SEQ_SAMPLE_INTERVAL_MS = 33; //~30 fps
const DYNAMIC_INFERENCE_INTERVAL_MS = 100;
const MIN_STABLE_SHAPE_COVERAGE = 0.6;
const MIN_STABLE_SHAPE_RATIO = 0.8;

//Voting
const VOTE_WINDOW = 16;
const HOLD_MS = 800;
const DYNAMIC_VOTE_WINDOW = 8;
const DYNAMIC_HOLD_MS = 500;
const CONF_THRESHOLD = 0.75;
const MIN_VOTE_SHARE = 0.6;
const MAX_MISSED_FRAMES = 3;
const IDLE_SPACE_MS = 10000;

//Recording
const RECORD_SAMPLE_COUNT = 32;
const RECORD_INTERVAL_MS = 90;
const READY_SAMPLE_COUNT = 20;

//REFERENCES
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const stage = document.getElementById("stage");

const statusBadge = document.getElementById("status-badge");
const predLabelEl = document.getElementById("pred-label");
const predConfEl = document.getElementById("pred-conf");
const holdBarEl = document.getElementById("hold-bar");
const guessLabelEl = document.getElementById("guess-label");
const outputText = document.getElementById("output-text");

const btnStart = document.getElementById("btn-start");
const btnFlip = document.getElementById("btn-flip");
const btnRecord = document.getElementById("btn-record");
const classSelect = document.getElementById("class-select");
const customClassInput = document.getElementById("custom-class");
const countsGrid = document.getElementById("counts-grid");

const drawingUtils = new DrawingUtils(ctx);

//APP State
let handLandmarker = null;
let dynamicSession = null;
let activeStream = null;
let facingMode = "user";
let loopRunning = false;
let lastVideoTime = -1;

let samples = loadSamples();
let classes = buildClassList(samples);
let recordRemaining = 0;
let lastRecordMs = 0;

//KNN cache
let knnIndex = { src: null, n: 0, data: new Float32Array(0), labels: [] };

//Dynamic sequence buffer
let rawSeqBuffer = [];
let rawSeqLabels = [];
let lastSeqSampleMs = 0;
let lastDynamicInferenceMs = 0;
let dynamicInferencePending = false;
let dynamicMotionActive = false;
let activePredictionMode = null;
let handTrackingEpoch = 0;

//Voting
let recentVotes = [];
let missedFrames = 0;
let currentStableLabel = null;
let holdStartMs = 0;
let committedThisHold = false;
let lastSignAt = null;

//Samples
function isValidSamples(list) {
  return Array.isArray(list) && list.every((sample) =>
    typeof sample?.label === "string" &&
    Array.isArray(sample.vec) &&
    sample.vec.length === FEATURE_DIM &&
    sample.vec.every(Number.isFinite));
}

function loadSamples() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      if (localStorage.getItem(LEGACY_STORAGE_KEY)) {
        console.info(
          "Hay muestras v1 sin migrar. Para migrarlas ejecuta " +
          `copy(localStorage.getItem("${LEGACY_STORAGE_KEY}")), pega el resultado en un .json e impórtalo.`
        );
      }
      return [];
    }
    const parsed = JSON.parse(raw);
    return parsed?.featureTag === FEATURE_TAG && isValidSamples(parsed.samples)
      ? parsed.samples
      : [];
  } catch {
    return [];
  }
}

function saveSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ featureTag: FEATURE_TAG, samples }));
  } catch (err) {
    console.warn("No se pudo guardar:", err);
    statusBadge.textContent = "No se pudo guardar (almacenamiento lleno). Exporta un respaldo.";
  }
  renderCounts();
}

function convertV1Vec(vec, width, height) {
  const points = new Array(FEATURE_DIM);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    points[3 * i] = vec[3 * i] * width;
    points[3 * i + 1] = vec[3 * i + 1] * height;
    points[3 * i + 2] = vec[3 * i + 2] * width;
  }
  let scale = 0; // the wrist is the origin of a v1 vector
  for (const idx of SCALE_REF_IDX) {
    scale += Math.hypot(points[3 * idx], points[3 * idx + 1], points[3 * idx + 2]);
  }
  scale = Math.max(scale / SCALE_REF_IDX.length, 1e-6);
  return points.map(roundTo4);
}

function samplesFromFile(data, legacySize) {
  if (data?.featureTag === FEATURE_TAG) {
    return isValidSamples(data.samples) ? data.samples : null;
  }
  if (Array.isArray(data) && STATIC_SPACE === "image" && legacySize) {
    try {
      const converted = data.map((s) => ({
        label: s.label,
        vec: convertV1Vec(s.vec, legacySize.width, legacySize.height)
      }));
      return isValidSamples(converted) ? converted : null;
    } catch {
      return null;
    }
  }
  return null;
}

function replaceSamples(list) {
  samples = list;
  classes = buildClassList(samples);
  renderClassDropdown();
  renderCounts();
}

function buildClassList(sampleList) {
  const set = new Set(DEFAULT_CLASSES);
  for (const s of sampleList) set.add(s.label);
  return Array.from(set);
}

//The Maths
function roundTo4(x) {
  return Math.round(x * 1e4) / 1e4;
}

function imagePoints(landmarks) {
  const W = video.videoWidth || 1;
  const H = video.videoHeight || 1;
  const out = new Float32Array(FEATURE_DIM);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = landmarks[i];
    out[3 * i] = p.x * W;
    out[3 * i + 1] = p.y * H;
    out[3 * i + 2] = p.z * W;
  }
  return out;
}

function worldPoints(worldLandmarks) {
  const out = new Float32Array(FEATURE_DIM);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = worldLandmarks[i];
    out[3 * i] = p.x;
    out[3 * i + 1] = p.y;
    out[3 * i + 2] = p.z;
  }
  return out;
}

function rawPoints(landmarks) {
  const out = new Float32Array(FEATURE_DIM);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = landmarks[i];
    out[3 * i] = p.x;
    out[3 * i + 1] = p.y;
    out[3 * i + 2] = p.z;
  }
  return out;
}

function palmScale(pts) {
  let sum = 0;
  for (const idx of SCALE_REF_IDX) {
    sum += Math.hypot(
      pts[3 * idx] - pts[0],
      pts[3 * idx + 1] - pts[1],
      pts[3 * idx + 2] - pts[2]
    );
  }
  return Math.max(sum / SCALE_REF_IDX.length, 1e-6);
}

function maxWristDistance(pts) {
  let max = 1e-6;
  for (let i = 1; i < NUM_LANDMARKS; i++) {
    const d = Math.hypot(pts[3 * i] - pts[0], pts[3 * i + 1] - pts[1], pts[3 * i + 2] - pts[2]);
    if (d > max) max = d;
  }
  return max;
}

function normalizePoints(pts) {
  const scale = palmScale(pts);
  const out = new Array(FEATURE_DIM);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[3 * i] = roundTo4((pts[3 * i] - pts[0]) / scale);
    out[3 * i + 1] = roundTo4((pts[3 * i + 1] - pts[1]) / scale);
    out[3 * i + 2] = roundTo4((pts[3 * i + 2] - pts[2]) / scale);
  }
  return out;
}

//Static Classifier (KNN)
function syncKnnIndex() {
  const ix = knnIndex;
  if (ix.src === samples && ix.n === samples.length) return;

  const grew = ix.src === samples && samples.length > ix.n;
  const start = grew ? ix.n : 0;
  const needed = samples.length * FEATURE_DIM;

  if (ix.data.length < needed) {
    const bigger = new Float32Array(Math.max(needed, ix.data.length * 2));
    if (grew) bigger.set(ix.data.subarray(0, ix.n * FEATURE_DIM));
    ix.data = bigger;
  }
  if (!grew) ix.labels = [];

  for (let i = start; i < samples.length; i++) {
    ix.data.set(samples[i].vec, i * FEATURE_DIM);
    ix.labels[i] = samples[i].label;
  }
  ix.src = samples;
  ix.n = samples.length;
}

function classifyKNN(vec, k = KNN_K) {
  syncKnnIndex();
  const { data, labels, n } = knnIndex;
  if (n === 0) return null;

  const kk = Math.min(k, n);
  const topD2 = new Float64Array(kk);
  const topIdx = new Int32Array(kk);
  let count = 0;

  for (let s = 0, base = 0; s < n; s++, base += FEATURE_DIM) {
    const worst = count === kk ? topD2[kk - 1] : Infinity;

    let d2 = 0;
    for (let i = 0; i < FEATURE_DIM; i++) {
      const d = vec[i] - data[base + i];
      d2 += d * d;
      if (d2 >= worst) break;
    }
    if (d2 >= worst) continue;

    let j = count < kk ? count : kk - 1;
    while (j > 0 && topD2[j - 1] > d2) {
      topD2[j] = topD2[j - 1];
      topIdx[j] = topIdx[j - 1];
      j--;
    }
    topD2[j] = d2;
    topIdx[j] = s;
    if (count < kk) count++;
  }

  const minDist = Math.sqrt(topD2[0]);
  if (minDist > MAX_MATCH_DIST) return { label: "?", confidence: 0, minDist };

  const weights = new Map();
  let totalWeight = 0;
  for (let j = 0; j < count; j++) {
    const w = 1 / (Math.sqrt(topD2[j]) + KNN_EPS);
    const label = labels[topIdx[j]];
    weights.set(label, (weights.get(label) || 0) + w);
    totalWeight += w;
  }

  let bestLabel = null;
  let bestWeight = -1;
  for (const [label, w] of weights) {
    if (w > bestWeight) {
      bestWeight = w;
      bestLabel = label;
    }
  }
  return { label: bestLabel, confidence: bestWeight / totalWeight, minDist };
}

// Dynamic Classifier (GRU)
const DYNAMIC_PROFILES = {
  legacy: { extractPoints: rawPoints, scaleOf: maxWristDistance, minMotion: 0.1, minWristTravel: 0.12 },
  v2: { extractPoints: imagePoints, scaleOf: palmScale, minMotion: 0.1, minWristTravel: 0.12 } // re-tune
};
if (!DYNAMIC_PROFILES[DYNAMIC_FEATURES]) {
  console.error(`DYNAMIC_FEATURES "${DYNAMIC_FEATURES}" no es válido; usando "legacy".`);
}
const DYNAMIC_PROFILE = DYNAMIC_PROFILES[DYNAMIC_FEATURES] || DYNAMIC_PROFILES.legacy;

function getMaxWristTravel(frames, scale) {
  const f0 = frames[0];
  let maxTravel = 0;
  for (let t = 1; t < frames.length; t++) {
    const f = frames[t];
    const travel = Math.hypot(f[0] - f0[0], f[1] - f0[1], f[2] - f0[2]) / scale;
    if (travel > maxTravel) maxTravel = travel;
  }
  return maxTravel;
}

function getMaxSequenceMotion(frames, scale) {
  const f0 = frames[0];
  let maxMotion = 0;
  for (let t = 1; t < frames.length; t++) {
    const f = frames[t];
    let sq = 0;
    for (let i = 0; i < FEATURE_DIM; i++) {
      const d = (f[i] - f0[i]) / scale;
      sq += d * d;
    }
    const motion = Math.sqrt(sq / NUM_LANDMARKS);
    if (motion > maxMotion) maxMotion = motion;
  }
  return maxMotion;
}

function getHandShapeStability(labels) {
  const knownLabels = labels.filter(Boolean);
  if (knownLabels.length < labels.length * MIN_STABLE_SHAPE_COVERAGE) return null;
  if (knownLabels[0] !== knownLabels[knownLabels.length - 1]) return false;

  const counts = new Map();
  for (const label of knownLabels) {
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  const mostCommonCount = Math.max(...counts.values());
  return mostCommonCount / knownLabels.length >= MIN_STABLE_SHAPE_RATIO;
}

function buildSequenceTensor(frames, scale) {
  const f0 = frames[0];
  const flat = new Float32Array(SEQ_LEN * FEATURE_DIM);
  let idx = 0;
  for (let t = 0; t < SEQ_LEN; t++) {
    const f = frames[t];
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      flat[idx++] = (f[3 * i] - f0[0]) / scale;
      flat[idx++] = (f[3 * i + 1] - f0[1]) / scale;
      flat[idx++] = (f[3 * i + 2] - f0[2]) / scale;
    }
  }
  return new ort.Tensor("float32", flat, [1, SEQ_LEN, FEATURE_DIM]);
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = Array.from(logits, (value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function pushSequenceFrame(points, staticLabel, now) {
  if (now - lastSeqSampleMs < SEQ_SAMPLE_INTERVAL_MS) return;
  rawSeqBuffer.push(points);
  rawSeqLabels.push(staticLabel);
  if (rawSeqBuffer.length > SEQ_LEN) rawSeqBuffer.shift();
  if (rawSeqLabels.length > SEQ_LEN) rawSeqLabels.shift();
  lastSeqSampleMs = now;
}

function detectDynamicMotion(scale) {
  const { minMotion, minWristTravel } = DYNAMIC_PROFILE;
  const hasMotion = getMaxSequenceMotion(rawSeqBuffer, scale) >= minMotion;
  const hasWristTravel = getMaxWristTravel(rawSeqBuffer, scale) >= minWristTravel;
  const shapeStable = getHandShapeStability(rawSeqLabels);
  return hasMotion && (shapeStable === true || (shapeStable === null && hasWristTravel));
}

async function runDynamicModel(scale) {
  const tensor = buildSequenceTensor(rawSeqBuffer, scale);
  const outputs = await dynamicSession.run({ [dynamicSession.inputNames[0]]: tensor });
  const probs = softmax(outputs[dynamicSession.outputNames[0]].data);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }
  return probs[bestIdx] < CONF_THRESHOLD
    ? null
    : { label: DYNAMIC_LABELS[bestIdx], confidence: probs[bestIdx] };
}

async function classifyDynamicBuffer(points, now, staticLabel) {
  if (!dynamicSession) return { prediction: null, motionActive: false };

  pushSequenceFrame(points, staticLabel, now);
  if (rawSeqBuffer.length < SEQ_LEN) {
    dynamicMotionActive = false;
    return { prediction: null, motionActive: false };
  }

  const scale = DYNAMIC_PROFILE.scaleOf(rawSeqBuffer[0]);
  dynamicMotionActive = detectDynamicMotion(scale);
  if (!dynamicMotionActive) return { prediction: null, motionActive: false };

  if (dynamicInferencePending || now - lastDynamicInferenceMs < DYNAMIC_INFERENCE_INTERVAL_MS) {
    return { prediction: null, motionActive: true };
  }

  dynamicInferencePending = true;
  lastDynamicInferenceMs = now;
  try {
    const prediction = await runDynamicModel(scale);
    return { prediction, motionActive: dynamicMotionActive };
  } finally {
    dynamicInferencePending = false;
  }
}

function resetDynamicState() {
  rawSeqBuffer = [];
  rawSeqLabels = [];
  lastSeqSampleMs = 0;
  dynamicMotionActive = false;
  activePredictionMode = null;
  handTrackingEpoch++;
}

//Predictions
function tallyVotes(latestLabel) {
  const tally = new Map();
  for (const vote of recentVotes) {
    const entry = tally.get(vote.label) ?? { n: 0, confSum: 0 };
    entry.n++;
    entry.confSum += vote.conf;
    tally.set(vote.label, entry);
  }

  let label = latestLabel;
  let best = tally.get(latestLabel);
  for (const [l, entry] of tally) {
    if (entry.n > best.n) {
      label = l;
      best = entry;
    }
  }
  return { label, share: best.n / recentVotes.length, avgConf: best.confSum / best.n };
}

function updatePredictionUI(label, conf, now, isDynamic = false) {
  missedFrames = 0;

  const voteWindow = isDynamic ? DYNAMIC_VOTE_WINDOW : VOTE_WINDOW;
  const minVotes = isDynamic ? 2 : Math.ceil(voteWindow / 2);

  recentVotes.push({ label, conf });
  while (recentVotes.length > voteWindow) recentVotes.shift();
  guessLabelEl.textContent = `Cadena de pensamiento: ${recentVotes.map((v) => v.label).join(" → ")}`; //Larp Larp Larp Sahur

  const winner = tallyVotes(label);
  const stable =
    recentVotes.length >= minVotes &&
    winner.share >= MIN_VOTE_SHARE &&
    winner.avgConf >= CONF_THRESHOLD;

  predLabelEl.textContent = winner.label;
  predConfEl.textContent = `Confianza: ${Math.round(winner.avgConf * 100)}%`;

  if (stable) {
    holdAndCommit(winner.label, now, isDynamic ? DYNAMIC_HOLD_MS : HOLD_MS);
  } else if (!committedThisHold) {
    currentStableLabel = null;
    holdBarEl.style.width = "0%";
  }
}

function holdAndCommit(label, now, holdDuration) {
  if (label !== currentStableLabel) {
    currentStableLabel = label;
    holdStartMs = now;
    committedThisHold = false;
  }
  const elapsed = now - holdStartMs;
  const pct = committedThisHold ? 100 : Math.min(100, (elapsed / holdDuration) * 100);
  holdBarEl.style.width = `${pct}%`;

  if (!committedThisHold && elapsed >= holdDuration) {
    outputText.value += label.length > 1 ? ` ${label} ` : label;
    lastSignAt = now;
    committedThisHold = true;
  }
}

function resetHoldUI(symbol) {
  recentVotes = [];
  currentStableLabel = null;
  committedThisHold = false;
  missedFrames = 0;
  predLabelEl.textContent = symbol;
  predConfEl.textContent = "Confianza: 0%";
  holdBarEl.style.width = "0%";
}

function registerMiss(symbol) {
  if (++missedFrames > MAX_MISSED_FRAMES) resetHoldUI(symbol);
}

function appendIdleSpace(now) {
  if (lastSignAt === null || now - lastSignAt < IDLE_SPACE_MS) return;
  if (!outputText.value.endsWith(" ")) outputText.value += " ";
  lastSignAt = null;
  predLabelEl.textContent = "- Espacio -";
}

function speakSpanish(text) {
  const clean = text.trim();
  if (!clean || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(clean);
  utter.lang = "es-MX";
  const voices = window.speechSynthesis.getVoices();
  const mxVoice = voices.find((v) => v.lang.toLowerCase().includes("es-mx"))
    || voices.find((v) => v.lang.toLowerCase().startsWith("es"));
  if (mxVoice) utter.voice = mxVoice;
  window.speechSynthesis.speak(utter);
}

//Frames
async function processFrame(now) {
  if (!loopRunning) return;
  requestAnimationFrame(processFrame);

  const hasNewFrame = handLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime;
  if (hasNewFrame) {
    lastVideoTime = video.currentTime;
    syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const result = handLandmarker.detectForVideo(video, performance.now());
    if (result.landmarks && result.landmarks.length > 0) {
      await handleHandFrame(result, now);
    } else {
      handleNoHand();
    }
  }
  appendIdleSpace(now);
}

function syncCanvasSize() {
  if (canvas.width === video.videoWidth && canvas.height === video.videoHeight) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
}

function drawHand(landmarks) {
  drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
    color: "#ffffff",
    lineWidth: 2
  });
  drawingUtils.drawLandmarks(landmarks, { color: "#008600", radius: 4 });
}

async function handleHandFrame(result, now) {
  const landmarks = result.landmarks[0];
  drawHand(landmarks);

  const staticPoints = STATIC_SPACE === "world"
    ? worldPoints(result.worldLandmarks[0])
    : imagePoints(landmarks);
  const vec = normalizePoints(staticPoints);

  recordSampleIfActive(vec, now);

  const pred = classifyKNN(vec);
  const epoch = handTrackingEpoch;
  try {
    const staticLabel = pred && pred.label !== "?" ? pred.label : null;
    const dynamicPoints = DYNAMIC_PROFILE.extractPoints(landmarks);
    const dynamicResult = await classifyDynamicBuffer(dynamicPoints, now, staticLabel);
    if (epoch !== handTrackingEpoch) return; // the hand left while the model was running
    applyPrediction(pred, dynamicResult, now);
  } catch (err) {
    console.error("Error al clasificar la seña dinámica:", err);
    statusBadge.textContent = "Error al clasificar la seña dinámica: " + err.message;
    resetHoldUI("—");
  }
}

function handleNoHand() {
  resetDynamicState();
  resetHoldUI("—");
}

//Static or not
function applyPrediction(pred, dynamicResult, now) {
  const mode = dynamicResult.motionActive ? "motion" : "idle";
  if (mode !== activePredictionMode) {
    activePredictionMode = mode;
    resetHoldUI(mode === "motion" ? "—" : "?");
  }

  if (mode === "motion") {
    const dynamicPred = dynamicResult.prediction;
    if (!dynamicPred) return;
    if (dynamicPred.label === "STATIC_IDLE") resetHoldUI("—");
    else updatePredictionUI(dynamicPred.label, dynamicPred.confidence, now, true);
  } else if (pred && pred.label !== "?") {
    updatePredictionUI(pred.label, pred.confidence, now);
  } else {
    registerMiss("?");
  }
}

function recordSampleIfActive(vec, now) {
  if (recordRemaining <= 0 || now - lastRecordMs < RECORD_INTERVAL_MS) return;

  const label = classSelect.value;
  samples.push({ label, vec });
  recordRemaining--;
  lastRecordMs = now;
  statusBadge.textContent = `Grabando ${label}: faltan ${recordRemaining}`;

  if (recordRemaining === 0) {
    saveSamples();
    statusBadge.textContent = `Listo (${samples.length} muestras)`;
  }
}

//Camera and models
async function initMediaPipe() {
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 1
      });
    } catch {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "VIDEO",
        numHands: 1
      });
    }
    statusBadge.textContent = "Modelo listo. Inicia la cámara.";
  } catch (err) {
    statusBadge.textContent = "Error al cargar MediaPipe: " + err.message;
  }
}

async function initDynamicModel() {
  try {
    dynamicSession = await ort.InferenceSession.create(DYNAMIC_MODEL_URL, {
      executionProviders: ["wasm"]
    });
  } catch (err) {
    console.warn("Dynamic model unavailable:", err);
    statusBadge.textContent = "Modelo dinámico no disponible (solo señas estáticas)";
  }
}

async function preloadBaseline() {
  if (samples.length > 0) return;
  try {
    const res = await fetch(BASELINE_URL);
    if (!res.ok) return;
    const data = await res.json();

    const list = samplesFromFile(data, LEGACY_CAPTURE_SIZE);
    if (!list || list.length === 0) {
      console.warn(`Base ignorada: se esperaba ${FEATURE_TAG}, llegó ${data?.featureTag ?? "sin etiqueta"}`);
      return;
    }
    if (Array.isArray(data)) {
      console.warn(
        `Base sin etiqueta de versión: convertida asumiendo ${LEGACY_CAPTURE_SIZE.width}x${LEGACY_CAPTURE_SIZE.height}. ` +
        "Para mayor precisión regenérala con lsm_features.py."
      );
    }
    if (samples.length > 0) return;

    replaceSamples(list);
    statusBadge.textContent = `Base cargada (${samples.length} muestras)`;
  } catch (err) {
    console.warn("No se pudo precargar la base:", err);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    statusBadge.textContent = "Cámara no disponible (requiere HTTPS o localhost).";
    return;
  }
  if (activeStream) {
    activeStream.getTracks().forEach((t) => t.stop());
  }
  stage.classList.toggle("mirrored", facingMode === "user");

  try {
    activeStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = activeStream;
    video.setAttribute("playsinline", "true");
    await video.play();

    lastVideoTime = -1;
    resetDynamicState();

    btnFlip.disabled = false;
    btnRecord.disabled = false;
    statusBadge.textContent = samples.length
      ? `Listo (${samples.length} muestras cargadas)`
      : "Cámara activa. Graba tus primeras señas abajo.";

    if (!loopRunning) {
      loopRunning = true;
      requestAnimationFrame(processFrame);
    }
  } catch (err) {
    statusBadge.textContent = "Permiso de cámara denegado: " + err.message;
  }
}

//UI Rendering
function renderClassDropdown(selected) {
  classSelect.innerHTML = "";
  for (const label of classes) {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    classSelect.appendChild(opt);
  }
  if (selected && classes.includes(selected)) classSelect.value = selected;
}

function renderCounts() {
  const counts = {};
  for (const c of classes) counts[c] = 0;
  for (const s of samples) counts[s.label] = (counts[s.label] || 0) + 1;

  countsGrid.innerHTML = "";
  for (const c of classes) {
    const n = counts[c] || 0;
    const pill = document.createElement("span");
    pill.className = "count-pill" + (n >= READY_SAMPLE_COUNT ? " ready" : "");
    pill.textContent = `${c}: ${n}`;
    pill.onclick = () => { classSelect.value = c; };
    countsGrid.appendChild(pill);
  }
}

//Event Handlers
function exportSamples() {
  const blob = new Blob(
    [JSON.stringify({ featureTag: FEATURE_TAG, samples })],
    { type: "application/json" }
  );
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "lsm_landmarks_backup_v2.json";
  a.click();
}

async function importSamples(file) {
  try {
    const imported = JSON.parse(await file.text());

    let legacySize = null;
    if (Array.isArray(imported)) {
      const answer = prompt(
        "Respaldo antiguo (v1). ¿Con qué resolución de video se grabó? Ejemplo: 640x480",
        `${LEGACY_CAPTURE_SIZE.width}x${LEGACY_CAPTURE_SIZE.height}`
      );
      const match = answer && answer.match(/^\s*(\d+)\s*x\s*(\d+)\s*$/i);
      if (!match) return;
      legacySize = { width: +match[1], height: +match[2] };
    }

    const list = samplesFromFile(imported, legacySize);
    if (!list) {
      statusBadge.textContent = "Archivo inválido o de otra versión de características";
      return;
    }
    replaceSamples(list);
    saveSamples();
  } catch (err) {
    statusBadge.textContent = "No se pudo importar: " + err.message;
  }
}

function addCustomClass() {
  const val = customClassInput.value.trim().toUpperCase();
  if (!val) return;
  if (!classes.includes(val)) classes.push(val);
  customClassInput.value = "";
  renderClassDropdown(val);
  renderCounts();
}

function deleteSelectedClass() {
  const target = classSelect.value;
  if (!confirm(`¿Borrar todas las muestras de "${target}"?`)) return;
  samples = samples.filter((s) => s.label !== target);
  saveSamples();
}

function resetAllSamples() {
  if (!confirm("¿Borrar todas las muestras y clases personalizadas? Esta acción no se puede deshacer.")) return;
  samples = [];
  saveSamples();
}

function wireEvents() {
  //Camera and recording
  btnStart.onclick = startCamera;
  btnFlip.onclick = () => {
    facingMode = facingMode === "user" ? "environment" : "user";
    startCamera();
  };
  btnRecord.onclick = () => {
    recordRemaining = RECORD_SAMPLE_COUNT;
    lastRecordMs = 0;
  };

  //Classes and samples
  document.getElementById("btn-add-class").onclick = addCustomClass;
  document.getElementById("btn-delete-class").onclick = deleteSelectedClass;
  document.getElementById("btn-reset-all").onclick = resetAllSamples;
  document.getElementById("btn-export").onclick = exportSamples;
  document.getElementById("input-import").onchange = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) await importSamples(file);
  };

  document.getElementById("btn-space").onclick = () => { outputText.value += " "; };
  document.getElementById("btn-back").onclick = () => { outputText.value = outputText.value.slice(0, -1); };
  document.getElementById("btn-clear").onclick = () => {
    outputText.value = "";
    lastSignAt = null;
  };
  document.getElementById("btn-speak").onclick = () => speakSpanish(outputText.value);
}

//Startup Finally
renderClassDropdown();
renderCounts();
wireEvents();
Promise.allSettled([initMediaPipe(), initDynamicModel(), preloadBaseline()]);