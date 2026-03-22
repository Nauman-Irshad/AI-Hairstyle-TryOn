/**
 * HairLab frontend — health checks, dropzone, and exhaustive error reporting.
 */

const $ = (id) => document.getElementById(id);

const HEALTH_INTERVAL_MS = 15000;
const REMOVE_BG_TIMEOUT_MS = 120000;
const REMOVE_HAIR_TIMEOUT_MS = 420000;

const HAIR_PLACEHOLDER_TITLE_DEFAULT = "Not available";
const HAIR_PLACEHOLDER_HINT_HTML =
  "Set <code>SKIP_HAIR_PIPELINE=0</code> and restart the server to load BiSeNet + LaMa for this panel.";

function setHealthUI(state, label, tooltip) {
  const pulse = $("healthPulse");
  const text = $("healthLabel");
  const wrap = $("headerStatus");
  pulse.classList.remove("ok", "warn", "bad");
  if (state === "ok") {
    pulse.classList.add("ok");
  } else if (state === "warn") {
    pulse.classList.add("warn");
  } else {
    pulse.classList.add("bad");
  }
  text.textContent = label;
  if (wrap) wrap.title = tooltip || label || "";
}

/**
 * Fetch /health with timeout; returns { ok, data, errorType, message, raw }
 */
async function checkHealth() {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 8000);
  try {
    const res = await fetch("/health", {
      method: "GET",
      cache: "no-store",
      signal: ctrl.signal,
    });
    clearTimeout(t);
    const ct = res.headers.get("content-type") || "";
    let bodyText = await res.text();
    let json = null;
    if (ct.includes("application/json")) {
      try {
        json = JSON.parse(bodyText);
      } catch {
        /* ignore */
      }
    }
    if (!res.ok) {
      return {
        ok: false,
        data: json,
        errorType: "HTTP_ERROR",
        message: `Health endpoint returned ${res.status} ${res.statusText}`,
        raw: bodyText.slice(0, 4000),
      };
    }
    if (!json) {
      return {
        ok: false,
        data: null,
        errorType: "BAD_RESPONSE",
        message: "Health response was not JSON (is this the FastAPI app?)",
        raw: bodyText.slice(0, 4000),
      };
    }
    const ready =
      json.ok === true &&
      (json.models_loaded === true || json.mode === "remove_bg_only");
    return {
      ok: ready,
      data: json,
      errorType: ready ? null : "MODELS_NOT_READY",
      message: ready
        ? null
        : json.message || "API is up but not ready.",
      raw: JSON.stringify(json, null, 2),
    };
  } catch (e) {
    clearTimeout(t);
    const name = e.name === "AbortError" ? "TIMEOUT" : "NETWORK";
    const origin = typeof window !== "undefined" ? window.location.origin : "";
    const msg =
      name === "TIMEOUT"
        ? "Request to /health timed out (server busy or blocked)."
        : `Cannot reach ${origin || "this page"}/health (${e.message || e}). Start the API with python run.py and open the exact URL it prints — if port 8000 is busy the server uses 8001, 8002, … so do not use a different port in the address bar.`;
    return {
      ok: false,
      data: null,
      errorType: name,
      message: msg,
      raw: String(e && e.stack ? e.stack : e),
    };
  }
}

function formatHttpError(res, bodyText, parsedJson) {
  const lines = [];
  lines.push(`HTTP ${res.status} ${res.statusText}`);
  lines.push(`URL: ${res.url || "(same origin)"}`);
  const ct = res.headers.get("content-type");
  if (ct) lines.push(`Content-Type: ${ct}`);

  if (parsedJson && typeof parsedJson === "object") {
    if (parsedJson.detail !== undefined) {
      const d = parsedJson.detail;
      if (Array.isArray(d)) {
        lines.push("detail (validation):");
        d.forEach((item, i) => {
          if (typeof item === "object" && item !== null) {
            lines.push(
              `  [${i}] ${item.loc ? item.loc.join(".") : "?"} — ${item.msg || JSON.stringify(item)}`,
            );
          } else {
            lines.push(`  [${i}] ${String(item)}`);
          }
        });
      } else {
        lines.push(`detail: ${typeof d === "string" ? d : JSON.stringify(d)}`);
      }
    } else {
      lines.push(`body: ${JSON.stringify(parsedJson, null, 2)}`);
    }
  } else if (bodyText) {
    lines.push("body (raw):");
    lines.push(bodyText.slice(0, 8000));
  } else {
    lines.push("(empty response body)");
  }
  return lines.join("\n");
}

function showErrorPanel(title, summary, technical) {
  $("errorSection").hidden = false;
  $("errorTitle").textContent = title;
  $("errorSummary").textContent = summary;
  $("errorTechnical").textContent = technical || "";
  $("errorDetailsWrap").open = true;
  $("errorSection").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideErrorPanel() {
  $("errorSection").hidden = true;
}

function clearErrorPanel() {
  hideErrorPanel();
}

async function getCameraStream() {
  const tries = [
    { video: true, audio: false },
    {
      video: { facingMode: { ideal: "user" }, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    },
  ];
  let last = null;
  for (const c of tries) {
    try {
      return await navigator.mediaDevices.getUserMedia(c);
    } catch (e) {
      last = e;
    }
  }
  throw last || new Error("getUserMedia failed");
}

/** POST /remove-background — rembg only, returns transparent PNG */
async function runRemoveBackground(formData) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REMOVE_BG_TIMEOUT_MS);
  let res;
  try {
    res = await fetch("/remove-background", {
      method: "POST",
      body: formData,
      signal: ctrl.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    const technical =
      e.name === "AbortError" ? `Timeout after ${REMOVE_BG_TIMEOUT_MS / 1000}s\n${e}` : `${e}\n${e.stack || ""}`;
    throw {
      title: e.name === "AbortError" ? "Request timed out" : "Network error",
      summary:
        e.name === "AbortError"
          ? "Background removal took too long (large image or first-time model download)."
          : `Could not complete request: ${e.message || e}`,
      technical,
    };
  }
  clearTimeout(timer);

  const ct = res.headers.get("content-type") || "";
  const buf = await res.arrayBuffer();
  const bodyText = new TextDecoder("utf-8").decode(buf);

  if (!res.ok) {
    let parsed = null;
    if (ct.includes("application/json")) {
      try {
        parsed = JSON.parse(bodyText);
      } catch {
        parsed = null;
      }
    }
    const tech = formatHttpError(res, bodyText, parsed);
    const short =
      parsed && parsed.detail
        ? typeof parsed.detail === "string"
          ? parsed.detail
          : "Validation or server error — see technical details."
        : res.statusText || "Request failed.";
    throw {
      title: `Server error (${res.status})`,
      summary: short,
      technical: tech,
    };
  }

  if (!ct.startsWith("image/")) {
    throw {
      title: "Unexpected response",
      summary: `Expected image/png, got: ${ct || "unknown"}`,
      technical: bodyText.slice(0, 4000),
    };
  }

  return new Blob([buf], { type: ct || "image/png" });
}

/** POST /remove-hair — BiSeNet + LaMa (requires loaded models) */
async function runRemoveHair(formData) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REMOVE_HAIR_TIMEOUT_MS);
  let res;
  try {
    res = await fetch("/remove-hair", {
      method: "POST",
      body: formData,
      signal: ctrl.signal,
    });
  } catch (e) {
    clearTimeout(timer);
    const technical =
      e.name === "AbortError" ? `Timeout after ${REMOVE_HAIR_TIMEOUT_MS / 1000}s\n${e}` : `${e}\n${e.stack || ""}`;
    throw {
      title: e.name === "AbortError" ? "Request timed out" : "Network error",
      summary:
        e.name === "AbortError"
          ? "Hair removal took too long (GPU, first-time model download, or large image)."
          : `Could not complete request: ${e.message || e}`,
      technical,
    };
  }
  clearTimeout(timer);

  const ct = res.headers.get("content-type") || "";
  const buf = await res.arrayBuffer();
  const bodyText = new TextDecoder("utf-8").decode(buf);

  if (!res.ok) {
    let parsed = null;
    if (ct.includes("application/json")) {
      try {
        parsed = JSON.parse(bodyText);
      } catch {
        parsed = null;
      }
    }
    const tech = formatHttpError(res, bodyText, parsed);
    const short =
      parsed && parsed.detail
        ? typeof parsed.detail === "string"
          ? parsed.detail
          : "Validation or server error — see technical details."
        : res.statusText || "Request failed.";
    throw {
      title: `Server error (${res.status})`,
      summary: short,
      technical: tech,
    };
  }

  if (!ct.startsWith("image/")) {
    throw {
      title: "Unexpected response",
      summary: `Expected image/png, got: ${ct || "unknown"}`,
      technical: bodyText.slice(0, 4000),
    };
  }

  return new Blob([buf], { type: ct || "image/png" });
}

function init() {
  const form = $("form");
  const fileInput = $("image");
  const dropzone = $("dropzone");
  const pickBtn = $("pickFile");
  const fileName = $("fileName");
  const dropPreview = $("dropPreview");
  const previewImg = $("previewImg");
  const dropIcon = $("dropIcon");
  const faceVideo = $("faceVideo");
  const camSnapCanvas = $("camSnapCanvas");
  const camStart = $("camStart");
  const camCapture = $("camCapture");
  const camStop = $("camStop");
  const camHint = $("camHint");
  const outImg = $("out");
  const outHairRemoved = $("outHairRemoved");
  const hairRemovedPlaceholder = $("hairRemovedPlaceholder");
  const hairRemovedLoading = $("hairRemovedLoading");
  const hairPlaceholderTitle = $("hairPlaceholderTitle");
  const hairPlaceholderHint = $("hairPlaceholderHint");
  const imgBefore = $("imgBefore");
  const compareStrip = $("compareStrip");
  const outputActions = $("outputActions");
  const downloadResult = $("downloadResult");
  const downloadHairRemoved = $("downloadHairRemoved");
  const placeholder = $("resultPlaceholder");
  const statusEl = $("status");
  const jobProgress = $("jobProgress");
  const jobProgressFill = $("jobProgressFill");
  const jobProgressMeta = $("jobProgressMeta");
  const jobProgressBar = $("jobProgressBar");
  const submitBtn = $("submit");
  let hairPulseTimer = null;
  let beforeObjectUrl = null;
  let resultObjectUrl = null;
  let hairResultObjectUrl = null;
  let previewObjectUrl = null;
  let camStream = null;
  /** @type {File | null} */
  let capturedFile = null;

  function setCamHint(t) {
    if (camHint) camHint.textContent = t || "";
  }

  /** Progress out of 10 steps (each step = 10%). Pass step 0 to hide. */
  function setProgressTen(step, message) {
    if (!jobProgress || !jobProgressFill) {
      if (statusEl && message) statusEl.textContent = message;
      return;
    }
    if (step <= 0) {
      jobProgress.hidden = true;
      jobProgressFill.style.width = "0%";
      if (jobProgressMeta) jobProgressMeta.textContent = "0 / 10 · 0%";
      if (jobProgressBar) jobProgressBar.setAttribute("aria-valuenow", "0");
      if (statusEl && message) statusEl.textContent = message;
      return;
    }
    jobProgress.hidden = false;
    const s = Math.min(10, Math.max(0, Math.round(step)));
    const pct = s * 10;
    jobProgressFill.style.width = `${pct}%`;
    if (jobProgressMeta) jobProgressMeta.textContent = `${s} / 10 · ${pct}%`;
    if (jobProgressBar) jobProgressBar.setAttribute("aria-valuenow", String(s));
    if (statusEl && message) statusEl.textContent = message;
  }

  function clearHairPulse() {
    if (hairPulseTimer) {
      clearInterval(hairPulseTimer);
      hairPulseTimer = null;
    }
  }

  function updatePreview(file) {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
    if (file && file.type.startsWith("image/")) {
      previewObjectUrl = URL.createObjectURL(file);
      previewImg.src = previewObjectUrl;
      previewImg.alt = "Preview of your upload";
      dropPreview.hidden = false;
      if (dropIcon) dropIcon.classList.add("drop-icon--hidden");
    } else {
      previewImg.removeAttribute("src");
      dropPreview.hidden = true;
      if (dropIcon) dropIcon.classList.remove("drop-icon--hidden");
    }
  }

  const clockEl = $("footerClock");
  function tickClock() {
    if (clockEl) {
      clockEl.textContent = new Date().toLocaleString(undefined, {
        weekday: "short",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    }
  }
  tickClock();
  setInterval(tickClock, 1000);

  pickBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    capturedFile = null;
    fileName.textContent = f ? f.name : "Optional if you used the camera above";
    updatePreview(f || null);
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) {
      fileInput.files = e.dataTransfer.files;
      fileName.textContent = f.name;
      updatePreview(f);
    }
  });
  dropzone.addEventListener("click", (e) => {
    if (e.target === pickBtn || e.target.closest(".link-like")) return;
    fileInput.click();
  });

  $("errorDismiss").addEventListener("click", hideErrorPanel);

  camStart?.addEventListener("click", async () => {
    setCamHint("");
    if (!window.isSecureContext) {
      setCamHint("Camera needs http://127.0.0.1:PORT or https (not http://192.168.x.x).");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamHint("Camera not supported in this browser.");
      return;
    }
    try {
      camStream = await getCameraStream();
      if (faceVideo) faceVideo.srcObject = camStream;
      await faceVideo?.play();
      if (camStop) camStop.disabled = false;
      setCamHint("Camera on — click “Use this frame” when your face is in view.");
    } catch (e) {
      setCamHint(e?.message || "Could not open camera. Allow permission or close other apps using it.");
    }
  });

  camCapture?.addEventListener("click", () => {
    const w = faceVideo?.videoWidth || 0;
    const h = faceVideo?.videoHeight || 0;
    if (!w || !h) {
      setCamHint("Start the camera and wait for the preview.");
      return;
    }
    camSnapCanvas.width = w;
    camSnapCanvas.height = h;
    const ctx = camSnapCanvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(faceVideo, 0, 0, w, h);
    camSnapCanvas.toBlob(
      (blob) => {
        if (!blob) return;
        capturedFile = new File([blob], "camera.jpg", { type: "image/jpeg" });
        fileInput.value = "";
        fileName.textContent = "Camera capture (ready)";
        updatePreview(capturedFile);
        setCamHint("Frame saved. Click Remove background, or choose a file instead.");
      },
      "image/jpeg",
      0.92,
    );
  });

  camStop?.addEventListener("click", () => {
    camStream?.getTracks().forEach((t) => t.stop());
    camStream = null;
    if (faceVideo) faceVideo.srcObject = null;
    if (camStop) camStop.disabled = true;
    setCamHint("");
  });

  async function pingHealth() {
    const r = await checkHealth();
    if (r.ok) {
      const d = r.data;
      let label = "Online";
      if (d?.mode === "remove_bg_only") {
        label = "Online · remove background (fast)";
      } else if (d?.models_loaded) {
        label = `Online · models ready${d.device ? ` · ${d.device}` : ""}`;
      }
      setHealthUI("ok", label, d?.message || "");
    } else if (r.errorType === "MODELS_NOT_READY") {
      setHealthUI("warn", "API up · models not ready", r.message || "");
    } else {
      const tip = [
        r.message || "Backend not responding.",
        `Current page: ${typeof window !== "undefined" ? window.location.origin : ""}.`,
        "Run: python run.py from the project folder, then open the URL shown in the terminal (same host and port as this tab).",
      ]
        .filter(Boolean)
        .join(" ");
      setHealthUI("bad", "Offline / unreachable", tip);
    }
    return r;
  }

  pingHealth();
  setInterval(pingHealth, HEALTH_INTERVAL_MS);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrorPanel();

    const file = capturedFile || (fileInput.files && fileInput.files[0]);
    if (!file) {
      showErrorPanel("No photo", "Use the camera (Start → Use this frame) or upload a portrait file.", "");
      return;
    }

    setProgressTen(1, "Starting…");
    const health = await checkHealth();
    if (!health.ok) {
      setProgressTen(0, "");
      const tech = [
        health.message || "Unknown",
        "",
        health.raw || "",
        health.data ? JSON.stringify(health.data, null, 2) : "",
      ]
        .filter(Boolean)
        .join("\n");
      showErrorPanel(
        health.errorType === "NETWORK" || health.errorType === "TIMEOUT" ? "Backend unreachable" : "Backend / health check failed",
        health.message || "Fix the server or network, then retry.",
        tech,
      );
      return;
    }

    setProgressTen(2, "Backend ready — removing background…");
    const fd = new FormData();
    fd.append("image", file);
    fd.append("max_side", "1280");

    function resetHairPlaceholderCopy() {
      if (hairPlaceholderTitle) hairPlaceholderTitle.textContent = HAIR_PLACEHOLDER_TITLE_DEFAULT;
      if (hairPlaceholderHint) hairPlaceholderHint.innerHTML = HAIR_PLACEHOLDER_HINT_HTML;
    }

    setProgressTen(3, "Removing background (rembg)…");
    submitBtn.disabled = true;
    try {
      const blob = await runRemoveBackground(fd);
      setProgressTen(4, "Cutout ready. Preparing preview…");
      if (beforeObjectUrl) URL.revokeObjectURL(beforeObjectUrl);
      if (resultObjectUrl) URL.revokeObjectURL(resultObjectUrl);
      if (hairResultObjectUrl) URL.revokeObjectURL(hairResultObjectUrl);
      hairResultObjectUrl = null;

      beforeObjectUrl = URL.createObjectURL(file);
      imgBefore.src = beforeObjectUrl;
      resultObjectUrl = URL.createObjectURL(blob);
      outImg.onload = () => {};
      outImg.src = resultObjectUrl;
      downloadResult.href = resultObjectUrl;
      downloadResult.download = `hairlab-cutout-${Date.now()}.png`;
      compareStrip.hidden = false;
      outputActions.hidden = false;
      placeholder.classList.add("hidden");

      if (outHairRemoved) outHairRemoved.hidden = true;
      if (downloadHairRemoved) downloadHairRemoved.hidden = true;
      resetHairPlaceholderCopy();
      if (hairRemovedLoading) hairRemovedLoading.hidden = true;

      const canHair =
        health.data?.hair_removal_available === true || health.data?.models_loaded === true;
      if (canHair) {
        if (hairRemovedPlaceholder) hairRemovedPlaceholder.hidden = true;
        if (hairRemovedLoading) hairRemovedLoading.hidden = false;
        setProgressTen(5, "Queuing hair removal (BiSeNet + LaMa)…");
        const fdHair = new FormData();
        fdHair.append("image", file);
        fdHair.append("max_side", "1280");
        fdHair.append("remove_bg", "true");
        try {
          const hairMsg = "Removing hair — segmentation & inpainting (LaMa)…";
          setProgressTen(6, hairMsg);
          let hairStep = 6;
          clearHairPulse();
          hairPulseTimer = setInterval(() => {
            if (hairStep < 8) {
              hairStep += 1;
              setProgressTen(hairStep, hairMsg);
            }
          }, 14000);
          const hairBlob = await runRemoveHair(fdHair);
          clearHairPulse();
          hairResultObjectUrl = URL.createObjectURL(hairBlob);
          if (outHairRemoved) {
            outHairRemoved.onload = () => {};
            outHairRemoved.src = hairResultObjectUrl;
            outHairRemoved.hidden = false;
          }
          if (hairRemovedLoading) hairRemovedLoading.hidden = true;
          if (downloadHairRemoved) {
            downloadHairRemoved.href = hairResultObjectUrl;
            downloadHairRemoved.download = `hairlab-hair-removed-${Date.now()}.png`;
            downloadHairRemoved.hidden = false;
          }
          setProgressTen(9, "Hair pass done — finishing…");
          setProgressTen(10, "Done — compare below or download PNGs.");
        } catch (hairErr) {
          clearHairPulse();
          if (hairRemovedLoading) hairRemovedLoading.hidden = true;
          if (hairRemovedPlaceholder) hairRemovedPlaceholder.hidden = false;
          if (hairPlaceholderTitle) hairPlaceholderTitle.textContent = "Hair removal failed";
          if (hairPlaceholderHint) {
            hairPlaceholderHint.textContent =
              hairErr.summary || hairErr.message || "See error details below.";
          }
          if (hairErr.title) {
            showErrorPanel(hairErr.title, hairErr.summary || "", hairErr.technical || "");
          }
          setProgressTen(4, "Cutout saved (4/10). Hair removal failed — see details.");
        }
      } else {
        if (hairRemovedPlaceholder) hairRemovedPlaceholder.hidden = false;
        setProgressTen(10, "Done — cutout ready. Load models for the third panel (see Hair removed).");
      }
    } catch (err) {
      clearHairPulse();
      setProgressTen(0, "");
      if (err.title) {
        showErrorPanel(err.title, err.summary, err.technical);
      } else {
        showErrorPanel("Unexpected error", err.message || String(err), err.stack || "");
      }
      statusEl.textContent = "";
    } finally {
      submitBtn.disabled = false;
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
