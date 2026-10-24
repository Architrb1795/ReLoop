/**
 * ReLoop — Frontend Application Controller
 * Brand: ReLoop | Tagline: Waste. Reward. Repeat.
 *
 * Connects the two primary product interfaces (RVM Machine & Student App)
 * to the unified PyTorch ResNet50 + Open-Set OOD Rejection backend (`/api/predict`,
 * `/api/deposit`, `/api/state`, `/api/test-sample`).
 *
 * Strictly enforces:
 *   - No hardcoded random predictions or mock classifiers.
 *   - Unified backend inference for both Uploaded Images and Live Camera frames.
 *   - Immediate state reset on new upload, mode switch, or object change.
 *   - Separation between AI recognition and verified (simulated) deposit.
 */

const appState = {
  activeTab: "rvm",
  inputMode: "upload",
  currentPrediction: null,
  isInferring: false,
  cameraStream: null,
  autoScanTimer: null,
  autoScanEnabled: false,
  recentCameraHistory: [], // Used for 2-frame temporal stabilization in auto-scan
  lastFrameLumaSignature: null,
  student: {
    is_demo_account: true,
    points_balance: 1280,
    next_reward_target: 1600,
    waste_disposed_kg: 12.4,
    material_recovered_kg: 8.7,
    successful_disposals: 24,
    recent_activity: [
      {
        id: "tx_demo_01",
        item_label: "Plastic Bottle",
        category: "plastic",
        points: 20,
        weight_kg: 0.025,
        timestamp: "Demo Baseline • Verified RVM-01",
        status: "VERIFIED_DEMO",
      },
      {
        id: "tx_demo_02",
        item_label: "Paper Cup",
        category: "paper",
        points: 10,
        weight_kg: 0.015,
        timestamp: "Demo Baseline • Verified RVM-02",
        status: "VERIFIED_DEMO",
      },
      {
        id: "tx_demo_03",
        item_label: "Aluminium Can",
        category: "metal",
        points: 30,
        weight_kg: 0.018,
        timestamp: "Demo Baseline • Verified RVM-01",
        status: "VERIFIED_DEMO",
      },
    ],
  },
};

// ============================================================================
// 1. INITIALIZATION & HEALTH CHECK
// ============================================================================
window.addEventListener("DOMContentLoaded", () => {
  renderStudentDashboard();
  fetchBackendHealthAndState();
});

async function fetchBackendHealthAndState() {
  try {
    const [healthRes, stateRes] = await Promise.all([
      fetch("/api/health", { cache: "no-store" }),
      fetch("/api/state", { cache: "no-store" }),
    ]);

    if (healthRes.ok) {
      const health = await healthRes.json();
      const badge = document.getElementById("engineStatusBadge");
      if (badge && health.engine_ready) {
        badge.textContent = "ResNet50 + OOD Active";
        badge.className = "badge badge-lime";
      }
    }

    if (stateRes.ok) {
      const st = await stateRes.json();
      appState.student = st;
      renderStudentDashboard();
    }
  } catch (err) {
    const badge = document.getElementById("engineStatusBadge");
    if (badge) {
      badge.textContent = "Server Offline";
      badge.className = "badge badge-error";
    }
  }
}

// ============================================================================
// 2. NAVIGATION BETWEEN THE TWO PRIMARY PRODUCT TABS
// ============================================================================
function switchProductTab(tabKey) {
  appState.activeTab = tabKey === "student" ? "student" : "rvm";

  const isRvm = appState.activeTab === "rvm";
  document.getElementById("pane-rvm").classList.toggle("active", isRvm);
  document.getElementById("pane-student").classList.toggle("active", !isRvm);

  // Sync top navbar tabs
  const navRvm = document.getElementById("navTabRvm");
  const navStudent = document.getElementById("navTabStudent");
  navRvm.classList.toggle("active", isRvm);
  navRvm.setAttribute("aria-selected", String(isRvm));
  navStudent.classList.toggle("active", !isRvm);
  navStudent.setAttribute("aria-selected", String(!isRvm));

  // Sync workspace header tabs
  document.getElementById("wsTabRvm").classList.toggle("active", isRvm);
  document.getElementById("wsTabStudent").classList.toggle("active", !isRvm);

  const wsTitle = document.getElementById("workspaceTitle");
  if (wsTitle) {
    wsTitle.textContent = isRvm
      ? "ReLoop RVM Console — AI Waste Identification & Verified Segregation"
      : "ReLoop Student App — Verified Impact & Rewards Wallet";
  }

  // Stop live camera if user navigates away from RVM tab to clean up resources
  if (!isRvm && appState.cameraStream) {
    stopLiveCamera();
  }
}

function startEarningAction() {
  switchProductTab("rvm");
  const ws = document.getElementById("workspace");
  if (ws) ws.scrollIntoView({ behavior: "smooth" });
}

// ============================================================================
// 3. INPUT SOURCE SWITCHING (UPLOAD IMAGE VS LIVE CAMERA)
// ============================================================================
function setInputSourceMode(mode) {
  appState.inputMode = mode === "camera" ? "camera" : "upload";
  const isCam = appState.inputMode === "camera";

  document.getElementById("modeBtnUpload").classList.toggle("active", !isCam);
  document.getElementById("modeBtnCamera").classList.toggle("active", isCam);

  document.getElementById("uploadControlsGroup").style.display = isCam ? "none" : "flex";
  document.getElementById("cameraControlsGroup").style.display = isCam ? "flex" : "none";
  document.getElementById("cameraErrorNotice").style.display = "none";

  resetRvmInspection();

  if (isCam) {
    document.getElementById("emptyStateTitle").textContent =
      "Start the live camera to identify waste items";
    document.getElementById("emptyStateSub").textContent =
      "Uses the same calibrated ResNet50 model and unknown-object rejection pipeline as image upload.";
  } else {
    if (appState.cameraStream) {
      stopLiveCamera();
    }
    document.getElementById("emptyStateTitle").textContent =
      "Upload a waste photo or select a sample below";
    document.getElementById("emptyStateSub").textContent =
      "Supports JPG, PNG, and WEBP (up to 10 MB). Recognizes Cardboard, Glass, Metal, Paper, Plastic, and Residual Wrappers — and safely rejects unknown objects such as stones.";
  }
}

function resetRvmInspection() {
  appState.currentPrediction = null;
  appState.recentCameraHistory = [];

  const imgPreview = document.getElementById("uploadedImagePreview");
  const videoEl = document.getElementById("liveCameraVideo");
  const emptyState = document.getElementById("viewportEmptyState");
  const frame = document.getElementById("rvmViewportFrame");
  const fileInput = document.getElementById("fileUploadInput");

  if (fileInput) fileInput.value = "";
  if (imgPreview) {
    imgPreview.src = "";
    imgPreview.style.display = "none";
  }

  if (!appState.cameraStream) {
    if (videoEl) videoEl.style.display = "none";
    if (emptyState) emptyState.style.display = "flex";
    if (frame) frame.classList.remove("has-media");
  }

  renderStandbyDecisionPanel();
}

function renderStandbyDecisionPanel() {
  const badge = document.getElementById("resultStateBadge");
  const banner = document.getElementById("resultStateBanner");
  const kicker = document.getElementById("resultCategoryKicker");
  const heading = document.getElementById("resultMainHeading");
  const expl = document.getElementById("resultExplanationText");
  const confWrap = document.getElementById("confidenceMeterWrap");
  const topPredSec = document.getElementById("topPredictionsSection");

  badge.textContent = "Awaiting Item";
  badge.className = "badge badge-neutral";
  banner.className = "result-state-banner";
  kicker.textContent = "Machine Standby";
  heading.textContent = "Present an item to begin";
  expl.textContent =
    "Upload a waste image or use the live camera. ReLoop will classify supported waste items or safely reject unknown objects before actuating the sorting gate.";

  confWrap.style.display = "none";
  topPredSec.style.display = "none";

  document.getElementById("sortingActionText").textContent = "Gate Locked (Standby)";
  document.getElementById("sortingStreamText").textContent = "—";
  document.getElementById("eligiblePointsText").textContent = "0 ReLoop Points";
  document.getElementById("disposalGuidanceText").textContent =
    "Awaiting valid waste recognition.";
  document.getElementById("btnConfirmDeposit").disabled = true;
}

// ============================================================================
// 4. UPLOADED IMAGE CLASSIFICATION FLOW
// ============================================================================
function handleFileInputChange(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  // Validate MIME type & file size on client before upload
  const validTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
  if (!validTypes.includes(file.type.toLowerCase())) {
    renderErrorDecisionPanel(
      "Unsupported File Format",
      "Please upload a valid JPG, PNG, or WEBP image file."
    );
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    renderErrorDecisionPanel(
      "File Too Large",
      "Maximum supported image size is 10 MB. Please choose a smaller image."
    );
    return;
  }

  // Clear previous prediction immediately so stale results are never shown
  appState.currentPrediction = null;
  renderStandbyDecisionPanel();

  const reader = new FileReader();
  reader.onerror = () => {
    renderErrorDecisionPanel("File Read Error", "Could not read the selected image file.");
  };
  reader.onload = async (e) => {
    const dataUrl = e.target.result;
    showUploadedPreviewImage(dataUrl);
    await sendImageForPrediction(dataUrl, "upload", file.name);
  };
  reader.readAsDataURL(file);
}

function showUploadedPreviewImage(dataUrl) {
  if (appState.cameraStream) {
    stopLiveCamera();
  }
  const imgPreview = document.getElementById("uploadedImagePreview");
  const videoEl = document.getElementById("liveCameraVideo");
  const emptyState = document.getElementById("viewportEmptyState");
  const frame = document.getElementById("rvmViewportFrame");

  videoEl.style.display = "none";
  emptyState.style.display = "none";
  imgPreview.src = dataUrl;
  imgPreview.style.display = "block";
  frame.classList.add("has-media");
}

async function runBackendTestSample(sampleType) {
  if (appState.isInferring) return;
  setInputSourceMode("upload");
  setLoadingState(true, `Loading & evaluating ${sampleType.replace("_", " ")} sample...`);

  try {
    const res = await fetch(`/api/test-sample?type=${encodeURIComponent(sampleType)}`, {
      cache: "no-store",
    });
    const data = await res.json();
    if (!res.ok || data.status === "ERROR") {
      renderErrorDecisionPanel(
        "Sample Evaluation Error",
        data.reason || "Could not evaluate sample image."
      );
      return;
    }

    if (data.image_data_url) {
      showUploadedPreviewImage(data.image_data_url);
    }
    appState.currentPrediction = data;
    renderPredictionOutcome(data);
  } catch (err) {
    renderErrorDecisionPanel(
      "Network / Server Error",
      "Could not reach the ReLoop inference server. Ensure server.py is running."
    );
  } finally {
    setLoadingState(false);
  }
}

// ============================================================================
// 5. LIVE CAMERA DETECTION FLOW (WITH STABILIZATION & SCENE RESET)
// ============================================================================
async function startLiveCamera() {
  const errBox = document.getElementById("cameraErrorNotice");
  errBox.style.display = "none";

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    errBox.textContent =
      "Camera API is not supported in this browser environment. Please use Upload Image.";
    errBox.style.display = "block";
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });

    appState.cameraStream = stream;
    const videoEl = document.getElementById("liveCameraVideo");
    const imgPreview = document.getElementById("uploadedImagePreview");
    const emptyState = document.getElementById("viewportEmptyState");
    const frame = document.getElementById("rvmViewportFrame");

    imgPreview.style.display = "none";
    emptyState.style.display = "none";
    videoEl.srcObject = stream;
    videoEl.style.display = "block";
    frame.classList.add("has-media");

    document.getElementById("btnStartCamera").disabled = true;
    document.getElementById("btnCaptureFrame").disabled = false;
    document.getElementById("btnToggleAutoScan").disabled = false;
    document.getElementById("btnStopCamera").disabled = false;

    renderStandbyDecisionPanel();
    showReloopToast("Camera active. Present an item and click 'Identify Current Frame'.");
  } catch (err) {
    errBox.textContent =
      "Camera permission denied or no camera hardware detected. You can retry permission or switch to Upload Image.";
    errBox.style.display = "block";
  }
}

function stopLiveCamera() {
  if (appState.autoScanTimer) {
    clearInterval(appState.autoScanTimer);
    appState.autoScanTimer = null;
  }
  appState.autoScanEnabled = false;
  const autoBtn = document.getElementById("btnToggleAutoScan");
  if (autoBtn) autoBtn.textContent = "Auto-Scan: Off";

  if (appState.cameraStream) {
    appState.cameraStream.getTracks().forEach((t) => t.stop());
    appState.cameraStream = null;
  }

  const videoEl = document.getElementById("liveCameraVideo");
  const emptyState = document.getElementById("viewportEmptyState");
  const frame = document.getElementById("rvmViewportFrame");

  if (videoEl) {
    videoEl.srcObject = null;
    videoEl.style.display = "none";
  }
  if (emptyState) emptyState.style.display = "flex";
  if (frame) frame.classList.remove("has-media");

  document.getElementById("btnStartCamera").disabled = false;
  document.getElementById("btnCaptureFrame").disabled = true;
  document.getElementById("btnToggleAutoScan").disabled = true;
  document.getElementById("btnStopCamera").disabled = true;

  appState.recentCameraHistory = [];
  appState.lastFrameLumaSignature = null;
}

function toggleContinuousCameraScan() {
  if (!appState.cameraStream) return;
  appState.autoScanEnabled = !appState.autoScanEnabled;
  const autoBtn = document.getElementById("btnToggleAutoScan");

  if (appState.autoScanEnabled) {
    autoBtn.textContent = "Auto-Scan: On (1.8s)";
    autoBtn.classList.remove("btn-outline");
    autoBtn.classList.add("btn-navy");
    captureAndClassifyCameraFrame(true);
    appState.autoScanTimer = setInterval(() => {
      if (!appState.isInferring && appState.cameraStream) {
        captureAndClassifyCameraFrame(true);
      }
    }, 1800);
  } else {
    if (appState.autoScanTimer) {
      clearInterval(appState.autoScanTimer);
      appState.autoScanTimer = null;
    }
    autoBtn.textContent = "Auto-Scan: Off";
    autoBtn.classList.remove("btn-navy");
    autoBtn.classList.add("btn-outline");
  }
}

function computeFrameLumaSignature(ctx, w, h) {
  // 8x8 downsampled luminance grid to detect when an object enters/leaves the frame
  const data = ctx.getImageData(0, 0, w, h).data;
  const sig = [];
  const stepX = Math.floor(w / 8);
  const stepY = Math.floor(h / 8);
  for (let gy = 0; gy < 8; gy++) {
    for (let gx = 0; gx < 8; gx++) {
      const idx = (gy * stepY * w + gx * stepX) * 4;
      const luma = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      sig.push(luma);
    }
  }
  return sig;
}

function hasSceneChangedSignificantly(sigA, sigB) {
  if (!sigA || !sigB || sigA.length !== sigB.length) return true;
  let diffSum = 0;
  for (let i = 0; i < sigA.length; i++) {
    diffSum += Math.abs(sigA[i] - sigB[i]);
  }
  const meanDiff = diffSum / sigA.length;
  return meanDiff > 18.0; // Reset smoothing history if object changed/moved substantially
}

async function captureAndClassifyCameraFrame(isAutoLoop = false) {
  const videoEl = document.getElementById("liveCameraVideo");
  const canvas = document.getElementById("cameraCaptureCanvas");
  if (!videoEl || !appState.cameraStream || videoEl.readyState < 2) return;

  const vw = videoEl.videoWidth || 640;
  const vh = videoEl.videoHeight || 480;
  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, vw, vh);

  // Detect scene change to avoid stale predictions when an object is removed or replaced
  const currentSig = computeFrameLumaSignature(ctx, vw, vh);
  if (hasSceneChangedSignificantly(appState.lastFrameLumaSignature, currentSig)) {
    appState.recentCameraHistory = [];
  }
  appState.lastFrameLumaSignature = currentSig;

  const dataUrl = canvas.toDataURL("image/jpeg", 0.88);
  await sendImageForPrediction(dataUrl, "camera", "camera_frame.jpg", isAutoLoop);
}

// ============================================================================
// 6. UNIFIED BACKEND INFERENCE & TEMPORAL STABILIZATION
// ============================================================================
async function sendImageForPrediction(
  imageBase64,
  source,
  filename = "",
  isSilentAutoLoop = false
) {
  if (appState.isInferring) return;
  appState.isInferring = true;

  if (!isSilentAutoLoop) {
    setLoadingState(true, "Running calibrated ResNet50 & OOD verification...");
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: imageBase64,
        source: source,
        filename: filename,
      }),
    });

    const data = await res.json();
    if (!res.ok || data.status === "ERROR") {
      renderErrorDecisionPanel(
        "Identification Error",
        data.reason || "Unable to process image."
      );
      return;
    }

    // Apply 2-frame temporal stabilization in continuous camera mode to prevent flicker
    let finalResult = data;
    if (source === "camera" && isSilentAutoLoop) {
      appState.recentCameraHistory.push(data);
      if (appState.recentCameraHistory.length > 2) {
        appState.recentCameraHistory.shift();
      }
      if (appState.recentCameraHistory.length === 2) {
        const prev = appState.recentCameraHistory[0];
        const curr = appState.recentCameraHistory[1];
        if (
          prev.status === "RECOGNIZED" &&
          curr.status === "RECOGNIZED" &&
          prev.predicted_class !== curr.predicted_class
        ) {
          // Wait for consistent class confirmation across consecutive frames
          finalResult = {
            ...curr,
            status: "UNCERTAIN",
            recognized: false,
            acceptable_for_segregation: false,
            display_label: "Stabilizing Camera Prediction...",
            reason:
              "Object is moving or transitioning between categories. Hold item steady for 1 second.",
            sorting_action: "HOLD_FOR_STABILIZATION",
            reward_points_eligible: 0,
          };
        }
      }
    }

    appState.currentPrediction = finalResult;
    renderPredictionOutcome(finalResult);
  } catch (err) {
    renderErrorDecisionPanel(
      "Connection Error",
      "Could not reach the ReLoop inference server. Ensure the backend server is active."
    );
  } finally {
    appState.isInferring = false;
    setLoadingState(false);
  }
}

function setLoadingState(isLoading, message = "Analyzing item...") {
  const overlay = document.getElementById("viewportLoadingOverlay");
  const txt = document.getElementById("viewportLoadingText");
  if (txt && message) txt.textContent = message;
  if (overlay) overlay.classList.toggle("visible", isLoading);
}

// ============================================================================
// 7. RENDERING THE 5 DISTINCT RESULT STATES
// ============================================================================
function renderPredictionOutcome(result) {
  const badge = document.getElementById("resultStateBadge");
  const banner = document.getElementById("resultStateBanner");
  const kicker = document.getElementById("resultCategoryKicker");
  const heading = document.getElementById("resultMainHeading");
  const expl = document.getElementById("resultExplanationText");
  const confWrap = document.getElementById("confidenceMeterWrap");
  const confLbl = document.getElementById("confidenceLabelText");
  const confVal = document.getElementById("confidenceValueText");
  const confFill = document.getElementById("confidenceBarFill");
  const topPredSec = document.getElementById("topPredictionsSection");
  const topPredList = document.getElementById("topPredictionsList");
  const depositBtn = document.getElementById("btnConfirmDeposit");

  const status = result.status;

  if (status === "RECOGNIZED") {
    badge.textContent = "Recognized • Verified Class";
    badge.className = "badge badge-success";
    banner.className = "result-state-banner state-recognized";
    kicker.textContent = `Supported Waste Category • ${result.latency_ms} ms`;
    heading.textContent = result.display_label;
    expl.textContent = result.reason;

    const pct = (result.confidence * 100).toFixed(1);
    confWrap.style.display = "block";
    confLbl.textContent = "Calibrated Model Confidence";
    confVal.textContent = `${pct}%`;
    confFill.className = "progress-fill";
    confFill.style.width = `${Math.max(5, result.confidence * 100)}%`;

    document.getElementById("sortingActionText").textContent =
      `Route → ${result.bin_label}`;
    document.getElementById("sortingStreamText").textContent = result.sorting_stream;
    document.getElementById("eligiblePointsText").textContent =
      `+${result.reward_points_eligible} ReLoop Points`;
    document.getElementById("disposalGuidanceText").textContent =
      result.disposal_guidance;

    depositBtn.disabled = false;
  } else if (status === "UNKNOWN") {
    badge.textContent = "Rejected • Unknown Object";
    badge.className = "badge badge-unknown";
    banner.className = "result-state-banner state-unknown";
    kicker.textContent = `Out-of-Distribution Rejection • ${result.latency_ms} ms`;
    heading.textContent = "Unknown Object — Cannot Reliably Identify";
    expl.textContent = result.reason;

    confWrap.style.display = "block";
    confLbl.textContent = "Open-Set Knownness Score (Below Threshold)";
    const kScorePct = Math.max(0, Math.min(100, (result.knownness_score || 0) * 100)).toFixed(1);
    confVal.textContent = `${kScorePct}% (Rejected)`;
    confFill.className = "progress-fill purple";
    confFill.style.width = `${kScorePct}%`;

    document.getElementById("sortingActionText").textContent =
      "Safe Reject Compartment (Gate Locked)";
    document.getElementById("sortingStreamText").textContent =
      "Do Not Send to Recycling Bin";
    document.getElementById("eligiblePointsText").textContent = "0 ReLoop Points";
    document.getElementById("disposalGuidanceText").textContent =
      "Unsupported item rejected. Please remove the object from the intake chamber.";

    depositBtn.disabled = true;
  } else if (status === "UNCERTAIN") {
    badge.textContent = "Uncertain • Hold Gate";
    badge.className = "badge badge-warn";
    banner.className = "result-state-banner state-uncertain";
    kicker.textContent = `Ambiguous Prediction • ${result.latency_ms || 0} ms`;
    heading.textContent = result.display_label || "Uncertain Prediction";
    expl.textContent = result.reason;

    const pct = ((result.confidence || 0) * 100).toFixed(1);
    confWrap.style.display = "block";
    confLbl.textContent = "Top Class Confidence (Below Safe Threshold)";
    confVal.textContent = `${pct}%`;
    confFill.className = "progress-fill purple";
    confFill.style.width = `${pct}%`;

    document.getElementById("sortingActionText").textContent =
      "Gate Held — Reposition or Manual Review";
    document.getElementById("sortingStreamText").textContent =
      "Awaiting Clearer View";
    document.getElementById("eligiblePointsText").textContent = "0 ReLoop Points";
    document.getElementById("disposalGuidanceText").textContent =
      "Try a clearer image, better lighting, or rotate the item to expose its material profile.";

    depositBtn.disabled = true;
  } else if (status === "POOR_QUALITY") {
    badge.textContent = "Poor Image Quality";
    badge.className = "badge badge-error";
    banner.className = "result-state-banner state-poor";
    kicker.textContent = "Image Quality Check Failed";
    heading.textContent = "Poor Image Quality — Retry Required";
    expl.textContent = result.reason;

    confWrap.style.display = "none";

    document.getElementById("sortingActionText").textContent =
      "Gate Locked — No Actuation";
    document.getElementById("sortingStreamText").textContent = "—";
    document.getElementById("eligiblePointsText").textContent = "0 ReLoop Points";
    document.getElementById("disposalGuidanceText").textContent =
      "Provide a well-lit, focused image with the waste item clearly centered.";

    depositBtn.disabled = true;
  } else {
    renderErrorDecisionPanel("Recognition Error", result.reason || "Unexpected error.");
    return;
  }

  // Render Top-3 diagnostic probabilities when present and status is RECOGNIZED or UNCERTAIN
  if (
    result.top_predictions &&
    result.top_predictions.length > 0 &&
    (status === "RECOGNIZED" || status === "UNCERTAIN")
  ) {
    topPredSec.style.display = "block";
    topPredList.innerHTML = result.top_predictions
      .map((tp, idx) => {
        const p = (tp.probability * 100).toFixed(1);
        return `
          <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; padding:5px 10px; background:var(--bg-secondary); border-radius:6px; border:1px solid var(--border-light);">
            <span style="font-weight:${idx === 0 ? "700" : "500"}; color:var(--reloop-navy);">
              ${idx + 1}. ${tp.label}
            </span>
            <span style="font-weight:700; color:${idx === 0 ? "var(--reloop-navy)" : "var(--text-muted)"};">${p}%</span>
          </div>
        `;
      })
      .join("");
  } else {
    topPredSec.style.display = "none";
  }
}

function renderErrorDecisionPanel(title, reasonText) {
  const badge = document.getElementById("resultStateBadge");
  const banner = document.getElementById("resultStateBanner");
  const kicker = document.getElementById("resultCategoryKicker");
  const heading = document.getElementById("resultMainHeading");
  const expl = document.getElementById("resultExplanationText");

  badge.textContent = "Error";
  badge.className = "badge badge-error";
  banner.className = "result-state-banner state-error";
  kicker.textContent = "Input Validation / System Notice";
  heading.textContent = title;
  expl.textContent = reasonText;

  document.getElementById("confidenceMeterWrap").style.display = "none";
  document.getElementById("topPredictionsSection").style.display = "none";
  document.getElementById("btnConfirmDeposit").disabled = true;
}

// ============================================================================
// 8. VERIFIED (SIMULATED) DEPOSIT & WALLET SYNCHRONIZATION
// ============================================================================
async function confirmSimulatedDeposit() {
  const pred = appState.currentPrediction;
  if (!pred || pred.status !== "RECOGNIZED" || !pred.acceptable_for_segregation) {
    return;
  }

  const depositBtn = document.getElementById("btnConfirmDeposit");
  depositBtn.disabled = true;

  try {
    const res = await fetch("/api/deposit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prediction_id: pred.prediction_id }),
    });

    const data = await res.json();
    if (!res.ok || data.status !== "DEPOSITED") {
      showReloopToast(data.reason || "Could not verify deposit.");
      return;
    }

    appState.student = data.student_state;
    renderStudentDashboard();

    // Update RVM UI to reflect completed deposit
    document.getElementById("resultStateBadge").textContent = "Deposited & Verified";
    document.getElementById("sortingActionText").textContent =
      `Completed → ${pred.bin_label} (Simulated)`;

    showReloopToast(
      `Verified! +${data.transaction.points} ReLoop Points added for ${data.transaction.item_label}.`
    );
  } catch (err) {
    showReloopToast("Network error while recording simulated deposit.");
  }
}

// ============================================================================
// 9. STUDENT DASHBOARD & WALLET RENDERING
// ============================================================================
function renderStudentDashboard() {
  const st = appState.student;
  const ptsFormatted = Number(st.points_balance).toLocaleString();

  const navPts = document.getElementById("navPointsDisplay");
  if (navPts) navPts.textContent = `${ptsFormatted} pts`;

  const setTxt = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setTxt("studentPointsMetric", ptsFormatted);
  setTxt("studentDisposedKgMetric", `${Number(st.waste_disposed_kg).toFixed(1)} kg`);
  setTxt("studentRecoveredKgMetric", `${Number(st.material_recovered_kg).toFixed(1)} kg`);
  setTxt("studentDisposalCountMetric", String(st.successful_disposals));

  // Progress bar toward next reward milestone (1,600 pts -> 1280/1600 = 80%)
  const target = st.next_reward_target || 1600;
  const pct = Math.min(100, Math.round((st.points_balance / target) * 100));
  setTxt("rewardProgressPctText", `${pct}%`);
  const barFill = document.getElementById("rewardProgressBarFill");
  if (barFill) barFill.style.width = `${pct}%`;
  setTxt(
    "rewardProgressSubText",
    `${ptsFormatted} / ${Number(target).toLocaleString()} ReLoop Points (Demo baseline + verified session deposits)`
  );

  // Recent Activity Feed
  const listEl = document.getElementById("studentActivityList");
  const countBadge = document.getElementById("activityCountBadge");
  if (countBadge) {
    countBadge.textContent = `${st.recent_activity.length} Records`;
  }

  if (listEl) {
    listEl.innerHTML = st.recent_activity
      .slice(0, 8)
      .map((item) => {
        const isLive = item.status === "SIMULATED_VERIFIED";
        return `
          <div class="activity-item">
            <div>
              <div style="display:flex; align-items:center; gap:8px;">
                <strong style="font-size:0.92rem; color:var(--reloop-navy);">${item.item_label}</strong>
                <span class="badge ${isLive ? "badge-lime" : "badge-neutral"}" style="font-size:0.65rem; padding:2px 8px;">
                  ${isLive ? "Session Deposit" : "Demo Sample"}
                </span>
              </div>
              <div style="font-size:0.76rem; color:var(--text-muted); margin-top:2px;">
                ${item.timestamp}
              </div>
            </div>
            <span class="activity-pts">+${item.points} points</span>
          </div>
        `;
      })
      .join("");
  }
}

// ============================================================================
// 10. TOAST HELPER
// ============================================================================
let toastTimer = null;
function showReloopToast(message) {
  const toast = document.getElementById("reloopToast");
  const msgEl = document.getElementById("reloopToastMessage");
  if (!toast || !msgEl) return;

  msgEl.textContent = message;
  toast.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("visible");
  }, 4200);
}

function hideReloopToast() {
  const toast = document.getElementById("reloopToast");
  if (toast) toast.classList.remove("visible");
}
