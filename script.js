/* AI Daydreaming Detector — Prototype
   Uses MediaPipe FaceMesh + TensorFlow.js
   Author: ChatGPT (prototype)
*/

// ----------------- Configuration -----------------
const SAMPLE_WINDOW_SECONDS = 2.0; // compute features per window
const SAMPLE_STRIDE = 1.0; // seconds between windows when collecting features
const EAR_BLINK_THRESHOLD = 0.20; // initial EAR threshold for blink detection (tweak)
const BLINK_MIN_INTERVAL = 200; // ms minimal separation between blinks
const ONSET_PROB_THRESHOLD = 0.60; // prob to consider "daydreaming" onset
const ONSET_RATE_INCREASE = 0.08; // rising probability across last windows indicates onset
const MODEL_EPOCHS = 25;
const FEATURE_VECTOR_SIZE = 6; // meanEAR, stdEAR, blinkRate, gazeVar, rollMean, rollStd

// ----------------- DOM -----------------
const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const recordFocusBtn = document.getElementById("recordFocus");
const recordDaydreamBtn = document.getElementById("recordDaydream");
const stopRecordingBtn = document.getElementById("stopRecording");
const downloadDataBtn = document.getElementById("downloadData");
const uploadDataInput = document.getElementById("uploadData");
const sampleDurationInput = document.getElementById("sampleDuration");
const recordStatus = document.getElementById("recordStatus");

const trainBtn = document.getElementById("trainBtn");
const trainStatus = document.getElementById("trainStatus");
const saveModelBtn = document.getElementById("saveModelBtn");
const loadModelInput = document.getElementById("loadModel");

const startDetectBtn = document.getElementById("startDetect");
const stopDetectBtn = document.getElementById("stopDetect");
const liveStatus = document.getElementById("liveStatus");
const detLabel = document.getElementById("detLabel");
const dayProbElem = document.getElementById("dayProb");
const blinkRateElem = document.getElementById("blinkRate");
const gazeVarElem = document.getElementById("gazeVar");
const headRollElem = document.getElementById("headRoll");
const onsetAlert = document.getElementById("onsetAlert");

// ----------------- State -----------------
let faceMesh, camera;
let gathering = false;
let samples = []; // array of {features:[],label:0/1}
let liveLoop = null;
let model = null;
let featureHistory = []; // sliding recent feature vectors for onset detection
let probHistory = [];

let lastLabel = "Focused";
let daydreamTimer = null;
let soundUnlocked = false;



let predLabel = 0; // 0 = focused, 1 = daydream
const beepSound = document.getElementById("beepSound");
const deepWorkSound = document.getElementById("deepWorkSound");
const deepWorkBtn = document.getElementById("deepWorkBtn");

// For tracking focus loss
let focusLostTime = null;
let beepPlaying = false;

// Ensure beep loops correctly
beepSound.loop = true;

let deepWorkEnabled = false;

deepWorkBtn.addEventListener("click", () => {
  deepWorkEnabled = !deepWorkEnabled;

  if (deepWorkEnabled) {
    deepWorkSound.play();
    deepWorkBtn.textContent = "Stop Deep Work Music";
  } else {
    deepWorkSound.pause();
    deepWorkSound.currentTime = 0;
    deepWorkBtn.textContent = "Deep Work Music";
  }
});


document.getElementById("unlockSound").addEventListener("click", () => {
    const testAudio = new Audio("beep.mp3");

    testAudio.play().then(() => {
        testAudio.pause();
        testAudio.currentTime = 0;
        soundUnlocked = true;
        console.log("Sound unlocked");
        document.getElementById("unlockSound").style.display = "none";
    }).catch(err => console.log("Sound unlock failed:", err));
});

// // check if label changed
// if (predLabel !== lastLabel) {
//     lastLabel = predLabel;

//     // if became daydreaming
//     if (predLabel === "Daydreaming") {

//         // clear old timer
//         if (daydreamTimer) clearTimeout(daydreamTimer);

//         // start a new one
//         daydreamTimer = setTimeout(() => {
//             if (soundUnlocked) {
//                 playBeep();
//             } else {
//                 console.log("Sound blocked: user has not enabled sound.");
//             }
//         }, 3000);
//     }
//     else {
//         // clear timer if user returns to focused
//         if (daydreamTimer) clearTimeout(daydreamTimer);
//     }
// }



beepSound.pause();
beepSound.currentTime = 0;
beepPlaying = false;

deepWorkSound.pause();
deepWorkSound.currentTime = 0;
deepWorkEnabled = false;
deepWorkBtn.textContent = "Deep Work Music";

function playBeep() {
  if (!soundUnlocked) {
    console.log("Sound blocked: user has not enabled sound.");
    return;
  }
  try {
    // use the existing <audio id="beepSound"> element so the browser recognizes user-unlock
    beepSound.currentTime = 0;
    const playPromise = beepSound.play();
    if (playPromise && playPromise.catch) {
      playPromise.catch(err => console.log("Audio play blocked:", err));
    }
  } catch (e) {
    console.log("playBeep error", e);
  }
}





// ----------------- Utilities: landmarks indices -----------------
/* MediaPipe FaceMesh uses 468 landmarks; these are common approximate indices:
   - left eye outer corner: 33, inner corner: 133
   - left eye vertical upper/lower: 159, 145  (approx)
   - right eye outer corner: 362, inner corner: 263
   - right eye vertical upper/lower: 386, 374 (approx)
   - iris center indices (if available) around 468-473 (may not be present in face_mesh)
   We'll use eye corner landmarks and average centers to compute EAR and gaze proxy.
*/
const L = {
  left_outer: 33, left_inner: 133, left_upper: 159, left_lower: 145,
  right_outer: 362, right_inner: 263, right_upper: 386, right_lower: 374,
  left_eye_center: 168, right_eye_center: 393, // approximate center landmarks
  nose_tip: 1,
  left_eye_center_iris: 468, right_eye_center_iris: 473 // only if model returns iris
};

// simple dist
function d(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

// ----------------- Feature extraction -----------------
function computeEAR(landmarks, side='left'){
  // Eye Aspect Ratio approximation with four landmarks
  try {
    if (side === 'left'){
      const p1 = landmarks[L.left_outer];
      const p2 = landmarks[L.left_inner];
      const pu = landmarks[L.left_upper];
      const pl = landmarks[L.left_lower];
      const horizontal = d(p1,p2);
      const vertical = d(pu,pl);
      if (horizontal === 0) return 0;
      return vertical / horizontal;
    } else {
      const p1 = landmarks[L.right_outer];
      const p2 = landmarks[L.right_inner];
      const pu = landmarks[L.right_upper];
      const pl = landmarks[L.right_lower];
      const horizontal = d(p1,p2);
      const vertical = d(pu,pl);
      if (horizontal === 0) return 0;
      return vertical / horizontal;
    }
  } catch(e){
    return 0;
  }
}

function computeGazeProxy(landmarks, side='left'){
  // proxy: position of eye center relative to eye corners (0..1)
  try{
    if (side === 'left'){
      const outer = landmarks[L.left_outer];
      const inner = landmarks[L.left_inner];
      const center = landmarks[L.left_eye_center] || {x:(outer.x+inner.x)/2, y:(outer.y+inner.y)/2};
      // normalized x: 0 at outer, 1 at inner
      const horizontal = (center.x - outer.x) / (inner.x - outer.x || 0.0001);
      const vertical = (center.y - ((landmarks[L.left_upper].y+landmarks[L.left_lower].y)/2)) || 0;
      return {hx:horizontal, vy:vertical};
    } else {
      const outer = landmarks[L.right_outer];
      const inner = landmarks[L.right_inner];
      const center = landmarks[L.right_eye_center] || {x:(outer.x+inner.x)/2, y:(outer.y+inner.y)/2};
      const horizontal = (center.x - outer.x) / (inner.x - outer.x || 0.0001);
      const vertical = (center.y - ((landmarks[L.right_upper].y+landmarks[L.right_lower].y)/2)) || 0;
      return {hx:horizontal, vy:vertical};
    }
  } catch(e){
    return {hx:0.5, vy:0};
  }
}

function computeHeadRoll(landmarks){
  // approximate roll (tilt) from eyes line: angle in degrees
  try{
    const left = landmarks[L.left_eye_center];
    const right = landmarks[L.right_eye_center];
    const dx = right.x - left.x;
    const dy = right.y - left.y;
    const angle = Math.atan2(dy,dx) * 180 / Math.PI; // degrees
    return angle;
  }catch(e){
    return 0;
  }
}

// ----------------- Blink detection (simple EAR threshold crossing) -----------------
let lastBlinkTime = 0;
function detectBlink(currentEAR){
  const now = performance.now();
  if (currentEAR < EAR_BLINK_THRESHOLD && (now - lastBlinkTime) > BLINK_MIN_INTERVAL){
    lastBlinkTime = now;
    return true;
  }
  return false;
}

// ----------------- MediaPipe init -----------------
async function initFaceMesh(){
  faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  faceMesh.onResults(onResults);

  const videoElement = video;
  camera = new Camera(videoElement, {
    onFrame: async () => { await faceMesh.send({image: videoElement}); },
    width: 1280,
    height: 720
  });
  camera.start();
}

// render overlay small landmarks
function drawResults(landmarks){
  const w = overlay.width = video.clientWidth;
  const h = overlay.height = video.clientHeight;
  ctx.clearRect(0,0,w,h);
  if (!landmarks) return;
  ctx.fillStyle = "rgba(78,123,255,0.9)";
  for (let i=0;i<landmarks.length;i+=4){
    const x = landmarks[i].x * w;
    const y = landmarks[i].y * h;
    ctx.beginPath(); ctx.arc(x,y,1.6,0,Math.PI*2); ctx.fill();
  }
}

// buffers for computing windowed features
let windowBuffer = []; // array of {timestamp, leftEAR, rightEAR, gazeHx, gazeRy, roll, blink (0/1)}

function onResults(results){
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    drawResults(null);
    return;
  }
  const lm = results.multiFaceLandmarks[0];

  drawResults(lm);

  // scale coords to js friendly objects
  const landmarks = lm.map(p=>({x:p.x, y:p.y, z:p.z}));

  // compute features
  const leftEAR = computeEAR(landmarks,'left');
  const rightEAR = computeEAR(landmarks,'right');
  const meanEAR = (leftEAR + rightEAR) / 2.0;

  const leftG = computeGazeProxy(landmarks,'left');
  const rightG = computeGazeProxy(landmarks,'right');
  const gazeHx = (leftG.hx + rightG.hx)/2;
  const gazeVy = (leftG.vy + rightG.vy)/2;

  const roll = computeHeadRoll(landmarks);

  const blink = detectBlink(meanEAR) ? 1 : 0;

  // push to window buffer
  windowBuffer.push({
    t: performance.now(),
    leftEAR, rightEAR, meanEAR,
    gazeHx, gazeVy,
    roll,
    blink
  });

  // keep only last N seconds (say 10s)
  const maxKeep = 10000;
  const now = performance.now();
  windowBuffer = windowBuffer.filter(item => (now - item.t) <= maxKeep);
}

// ----------------- Feature vector builder (over WINDOW seconds) -----------------
function buildFeatureFromWindow(windowSeconds=SAMPLE_WINDOW_SECONDS){
  const now = performance.now();
  const cutoff = now - windowSeconds*1000;
  const segment = windowBuffer.filter(s => s.t >= cutoff);
  if (segment.length === 0) return null;

  const meanEAR = segment.reduce((a,b)=>a+b.meanEAR,0)/segment.length;
  const stdEAR = Math.sqrt(segment.reduce((a,b)=>a+Math.pow(b.meanEAR-meanEAR,2),0)/segment.length);
  // blink rate per minute
  const blinks = segment.reduce((a,b)=>a+b.blink,0);
  const durationMinutes = (segment[segment.length-1].t - segment[0].t) / 60000 || (windowSeconds/60.0);
  const blinkRate = blinks / durationMinutes;

  // gaze stability: variance of gazeHx and gazeVy (higher => unstable)
  const meanGx = segment.reduce((a,b)=>a+b.gazeHx,0)/segment.length;
  const gxVar = segment.reduce((a,b)=>a+Math.pow(b.gazeHx-meanGx,2),0)/segment.length;

  // head roll mean + std
  const meanRoll = segment.reduce((a,b)=>a+b.roll,0)/segment.length;
  const stdRoll = Math.sqrt(segment.reduce((a,b)=>a+Math.pow(b.roll-meanRoll,2),0)/segment.length);

  return [meanEAR, stdEAR, blinkRate, gxVar, meanRoll, stdRoll];
}

// ----------------- Data collection UI -----------------
let recordTimer = null;
function startRecording(label){
  if (gathering) return;
  gathering = true;
  const dur = Math.max(1, Number(sampleDurationInput.value) || 5);
  recordStatus.textContent = `Recording ${label===1?'Daydream':'Focus'} for ${dur}s...`;
  recordStatus.style.color = "";

  const start = performance.now();
  const sampleIntervalMs = 200; // collect features periodically
  const collected = [];

  recordTimer = setInterval(()=>{
    const feat = buildFeatureFromWindow(SAMPLE_WINDOW_SECONDS);
    if (feat) collected.push(feat);
  }, sampleIntervalMs);

  setTimeout(()=>{
    stopRecording();
    // store averaged feature vectors to dataset (we'll average windows to one vector for simplicity)
    if (collected.length > 0){
      // reduce to one representative feature: mean and std across collected windows
      const agg = collected[0].map((_,i)=> {
        const arr = collected.map(r=>r[i]);
        const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
        const std = Math.sqrt(arr.reduce((a,b)=>a+Math.pow(b-mean,2),0)/arr.length);
        return mean; // keep mean as feature
      });
      samples.push({features: agg, label: label}); // label: 0 focused, 1 daydream
      recordStatus.textContent = `Saved sample (${label===1?'Daydream':'Focus'}). Total samples: ${samples.length}`;
    } else {
      recordStatus.textContent = `No usable features captured. Try again.`;
    }
    gathering = false;
  }, dur*1000);
}

function stopRecording(){
  gathering = false;
  if (recordTimer) clearInterval(recordTimer);
  recordTimer = null;
  recordStatus.textContent = "Recording stopped";
}

// download dataset
downloadDataBtn.addEventListener('click', ()=>{
  const blob = new Blob([JSON.stringify(samples,null,2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'daydream_dataset.json';
  document.body.appendChild(a); a.click(); a.remove();
});

// upload dataset
uploadDataInput.addEventListener('change', async (ev)=>{
  const f = ev.target.files[0];
  if (!f) return;
  const text = await f.text();
  try{
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      samples = samples.concat(parsed);
      recordStatus.textContent = `Loaded ${parsed.length} samples. Total: ${samples.length}`;
    } else {
      alert('Invalid dataset format (expecting array).');
    }
  }catch(e){ alert('Error parsing JSON'); }
});

// buttons
recordFocusBtn.addEventListener('click', ()=> startRecording(0));
recordDaydreamBtn.addEventListener('click', ()=> startRecording(1));
stopRecordingBtn.addEventListener('click', stopRecording);

// ----------------- Model training (simple dense classifier) -----------------
async function trainModel(){
  if (samples.length < 6){
    trainStatus.textContent = "Need more samples (min ~6).";
    return;
  }
  trainStatus.textContent = "Preparing data...";
  // prepare tensors
  const X = tf.tensor2d(samples.map(s=>s.features));
  const y = tf.tensor1d(samples.map(s=>s.label),'int32');
  const yOneHot = tf.oneHot(y,2);

  // normalize features (compute mean/std on training set)
  const {mean, variance} = tf.moments(X,0);
  const std = tf.sqrt(variance);
  const Xn = X.sub(mean).div(std.add(1e-6));

  trainStatus.textContent = "Building model...";
  const m = tf.sequential();
  m.add(tf.layers.dense({units:32, activation:'relu', inputShape:[FEATURE_VECTOR_SIZE]}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.dense({units:16, activation:'relu'}));
  m.add(tf.layers.dense({units:2, activation:'softmax'}));

  m.compile({optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy', metrics:['accuracy']});

  trainStatus.textContent = "Training...";
  await m.fit(Xn, yOneHot, {
    epochs: MODEL_EPOCHS,
    batchSize: Math.min(32, samples.length),
    callbacks: {
      onEpochEnd: (e,logs) => {
        trainStatus.textContent = `Epoch ${e+1}/${MODEL_EPOCHS} — loss ${logs.loss.toFixed(3)} acc ${ (logs.acc || logs.acc).toFixed(3) }`;
      }
    }
  });

  // save mean/std with model for normalization
  model = {
    net: m,
    mean: await mean.array(),
    std: await std.array()
  };
  trainStatus.textContent = "Model trained.";
}

// save model (JSON + weights) downloaded with tfjs save
saveModelBtn.addEventListener('click', async ()=>{
  if (!model) { alert('No trained model.'); return; }
  trainStatus.textContent = "Saving model...";
  // save network
  await model.net.save('downloads://daydream_model');
  trainStatus.textContent = "Model saved (download). Note: normalization stats not saved automatically — download dataset to re-train or copy normalization.";
});

// load model weights (optional)
loadModelInput.addEventListener('change', async (ev)=>{
  const f = ev.target.files[0];
  if (!f) return;
  trainStatus.textContent = "Loading model from file...";
  try{
    // tfjs supports loadLayersModel from URL or file input via file:// is not trivial.
    // Simpler: instruct user to use model.json + shard files via hosted URL or drag-drop is complex.
    alert('Loading pretrained model via file input is not fully implemented in this prototype. Use "Train Model" or host model files.');
  }catch(e){
    console.error(e);
    trainStatus.textContent = "Load failed";
  }
});

trainBtn.addEventListener('click', ()=> trainModel());

// ----------------- Live detection -----------------
let detectInterval = null;
function startDetection(){
  if (!model) { liveStatus.textContent = "Train a model first."; return; }
  liveStatus.textContent = "Detecting...";
  onsetAlert.textContent = '';
  probHistory = [];
  featureHistory = [];

  detectInterval = setInterval(async ()=>{
    const feat = buildFeatureFromWindow(SAMPLE_WINDOW_SECONDS);
    if (!feat) return;
    // normalize
    const meanArr = model.mean;
    const stdArr = model.std;
    const norm = feat.map((v,i)=> (v - meanArr[i]) / (stdArr[i] + 1e-6));
    const pred = tf.tidy(()=> {
      const t = tf.tensor2d([norm]);
      const out = model.net.predict(t);
      return out.arraySync()[0]; // [pFocus, pDaydream]
    });
    const pDay = pred[1], pFocus = pred[0];
    probHistory.push(pDay);
    if (probHistory.length > 10) probHistory.shift();

    // compute simple metrics for display
    const recentFeature = feat;
    featureHistory.push({t:Date.now(), feat:recentFeature, prob:pDay});
    if (featureHistory.length > 30) featureHistory.shift();

    // compute blink rate and gaze variance display using current buffer
    const curBlinkRate = recentFeature[2];
    const curGazeVar = recentFeature[3];
    const curRoll = recentFeature[4];

    detLabel.textContent = pDay > 0.5 ? "Daydream" : "Focused";
    dayProbElem.textContent = pDay.toFixed(2);
    blinkRateElem.textContent = curBlinkRate.toFixed(1);
    gazeVarElem.textContent = curGazeVar.toFixed(4);
    headRollElem.textContent = curRoll.toFixed(1);

    // ---- label-change detection & 3s delayed beep ----
// use string labels for clarity
const currentLabel = (pDay > 0.5) ? "Daydreaming" : "Focused";

// if label changed since last check
if (currentLabel !== lastLabel) {
  // console.log("Label changed:", lastLabel, "->", currentLabel);
  lastLabel = currentLabel;

  // if it switched TO daydreaming, start 3s timer
  if (currentLabel === "Daydreaming") {
    if (daydreamTimer) clearTimeout(daydreamTimer);
    daydreamTimer = setTimeout(() => {
      // double-check current probability so transient blips won't trigger
      // use latest feature window to recompute if needed; here we use pDay
      if (pDay > 0.5) {
        playBeep();
      } else {
        // if probability dropped back during delay, don't beep
        // console.log("Daydreaming dropped during 3s delay; no beep.");
      }
    }, 30000);
  } else {
    // switched back to focused — cancel any pending beep and stop playing
    if (daydreamTimer) {
      clearTimeout(daydreamTimer);
      daydreamTimer = null;
    }
    // stop any currently playing beep audio
    try {
      if (!beepSound.paused) {
        beepSound.pause();
        beepSound.currentTime = 0;
      }
    } catch(e){/* ignore */}
  }
}


    // ONSET detection: rising probability + crossed threshold
    const len = probHistory.length;
    let onset = false;
    if (pDay >= ONSET_PROB_THRESHOLD && len >= 3){
      const rise = probHistory[len-1] - probHistory[0];
      if (rise > ONSET_RATE_INCREASE) onset = true;
    }
    if (onset){
      onsetAlert.textContent = `Early signs detected — possible onset (p=${pDay.toFixed(2)})`;
    } else {
      onsetAlert.textContent = '';
    }

  }, SAMPLE_STRIDE*1000);
}

function stopDetection(){
  if (detectInterval) clearInterval(detectInterval);
  liveStatus.textContent = "Idle";
  detLabel.textContent = "—";
  dayProbElem.textContent = "—";
  blinkRateElem.textContent = "—";
  gazeVarElem.textContent = "—";
  headRollElem.textContent = "—";
  onsetAlert.textContent = '';
}

startDetectBtn.addEventListener('click', startDetection);
stopDetectBtn.addEventListener('click', stopDetection);

// ----------------- Init camera & facemesh -----------------
async function init(){
  // fallback canvas sizing
  video.addEventListener('loadedmetadata', () => {
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
  });

  // get webcam access (try 720p)
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720}, audio:false});
    video.srcObject = stream;
  } catch (e){
    alert('Camera access denied or not available. This prototype requires camera access.');
    console.error(e);
  }

  await initFaceMesh();
  trainStatus.textContent = "FaceMesh running. Collect samples to train.";
}

init();

async function detectionLoop() {
    if (!faceMesh || !video) {
        requestAnimationFrame(detectionLoop);
        return;
    }

    // Run MediaPipe FaceMesh
    const predictions = await faceMesh.send({ image: video });

    if (predictions && predictions.multiFaceLandmarks && predictions.multiFaceLandmarks.length > 0) {
        const landmarks = predictions.multiFaceLandmarks[0];

        // Extract features for your ML model (example)
        const features = extractFeatures(landmarks);

        // Run your classifier to get predLabel
        const predLabel = await classifier.predict(features);

        // 🔔 ALERT IF DAYDREAMING
        if (predLabel === "Daydreaming") {
            console.log("⚠️ Daydream detected!");
            playBeep();
        }

        // Display on screen if needed
        document.getElementById("status").innerText = predLabel;
    }

    requestAnimationFrame(detectionLoop);
}

detectionLoop(); // ⬅ Start continuous detection loop

function extractFeatures(landmarks) {
    return landmarks.flatMap(p => [p.x, p.y, p.z]);
}




// ----------------- Helpful: expose samples for debugging -----------------
window._DAYDREAM_SAMPLES = samples;
window._MODEL = model;
