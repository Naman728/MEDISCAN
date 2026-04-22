(function () {
  const query = new URLSearchParams(window.location.search);
  const state = {
    modality: query.get("modality") || "brain-mri",
    config: null,
    sourceImage: "",
    comparisonMode: "side",
    isLoading: false,
    predictions: [],
    heatmaps: [],
    completionTimers: [],
    particleAnimationFrame: null,
    selectedFile: null
  };

  const els = {
    pageTitle: document.getElementById("pageTitle"),
    pageSubtitle: document.getElementById("pageSubtitle"),
    todayStamp: document.getElementById("todayStamp"),
    brandIcon: document.getElementById("brandIcon"),
    serviceLabel: document.getElementById("serviceLabel"),
    globalStatus: document.getElementById("globalStatus"),
    statusText: document.getElementById("statusText"),
    runAnalysisBtn: document.getElementById("runAnalysisBtn"),
    runBtnText: document.getElementById("runBtnText"),
    uploadDropzone: document.getElementById("uploadDropzone"),
    uploadInput: document.getElementById("scanUploadInput"),
    uploadFileName: document.getElementById("uploadFileName"),
    comparisonBody: document.getElementById("comparisonBody"),
    comparisonToggleGroup: document.getElementById("comparisonToggleGroup"),
    abnormalityList: document.getElementById("abnormalityList"),
    heatmapGrid: document.getElementById("heatmapGrid"),
    sourcePreview: document.getElementById("sourcePreview"),
    heatmapPreview: document.getElementById("heatmapPreview"),
    sourcePlaceholder: document.getElementById("sourcePlaceholder"),
    heatPlaceholder: document.getElementById("heatPlaceholder"),
    summaryHeader: document.getElementById("summaryHeader"),
    predictionCounter: document.getElementById("predictionCounter"),
    copyAllAbnormalitiesBtn: document.getElementById("copyAllAbnormalitiesBtn"),
    modelInfoList: document.getElementById("modelInfoList"),
    processingFeed: document.getElementById("processingFeed"),
    particleCanvas: document.getElementById("particleCanvas"),
    cursorDot: document.getElementById("cursorDot"),
    cursorRing: document.getElementById("cursorRing"),
    heatmapModal: document.getElementById("heatmapModal"),
    modalHeatmapImage: document.getElementById("modalHeatmapImage"),
    modalTitle: document.getElementById("modalTitle"),
    modalDownloadBtn: document.getElementById("modalDownloadBtn"),
    prevHeatmapBtn: document.getElementById("prevHeatmapBtn"),
    nextHeatmapBtn: document.getElementById("nextHeatmapBtn"),
    compareToggleBtn: document.getElementById("compareToggleBtn"),
    compareHeatmapSelect: document.getElementById("compareHeatmapSelect"),
    compareHeatmapImage: document.getElementById("compareHeatmapImage"),
    comparePane: document.getElementById("comparePane"),
    comparePlaceholder: document.getElementById("comparePlaceholder"),
    zoomInBtn: document.getElementById("zoomInBtn"),
    zoomOutBtn: document.getElementById("zoomOutBtn"),
    closeModalBtn: document.getElementById("closeModalBtn"),
    zoomStage: document.getElementById("zoomStage")
  };

  const modalState = {
    scale: 1,
    activeHeatmap: null,
    compareMode: false,
    compareHeatmap: null
  };


  const API_ENDPOINTS = {
    "brain-mri": "/predict/brain",
    xray: "/predict/lung-xray",
    "ct-scan": "/predict/lung-ct"
  };

  function getPredictionTargetCount() {
    const schemaCount = Array.isArray(state.config?.abnormalitySchema) ? state.config.abnormalitySchema.length : 0;
    return schemaCount || Math.max(state.predictions.length, 6);
  }

  function getHeatmapTargetCount() {
    const slotCount = Number(state.config?.heatmapSlots) || 0;
    const predictionCount = getPredictionTargetCount();
    return slotCount || predictionCount || Math.max(state.heatmaps.length, 6);
  }

  function formatDateStamp() {
    const now = new Date();
    const date = now.toLocaleDateString(undefined, {
      weekday: "short",
      year: "numeric",
      month: "short",
      day: "numeric"
    });
    const time = now.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit"
    });
    return `${date} • ${time}`;
  }

  function metricTemplate(title, value, subtitle, glow) {
    const copyPayload = `${title}: ${value} (${subtitle})`;
    return `
      <article class="metric-card surf-interactive" style="--metric-glow:${glow}">
        <div class="metric-top">
          <div class="metric-title">${title}</div>
          <button class="copy-btn surf-interactive" data-copy="${encodeURIComponent(copyPayload)}" type="button" aria-label="Copy ${title}">⧉</button>
        </div>
        <div class="metric-value">${value}</div>
        <div class="metric-subtitle">${subtitle}</div>
      </article>
    `;
  }

  function severityColor(score) {
    if (score >= 0.8) {
      return {
        from: "#ef4444",
        to: "#f97316",
        glow: "rgba(239,68,68,0.2)",
        label: "Critical"
      };
    }
    if (score >= 0.6) {
      return {
        from: "#f59e0b",
        to: "#f97316",
        glow: "rgba(245,158,11,0.22)",
        label: "Elevated"
      };
    }
    if (score >= 0.35) {
      return {
        from: "#4f8ef7",
        to: "#00b4d8",
        glow: "rgba(79,142,247,0.18)",
        label: "Moderate"
      };
    }
    return {
      from: "#22c55e",
      to: "#16a34a",
      glow: "rgba(34,197,94,0.2)",
      label: "Low"
    };
  }

  function clearCompletionTimers() {
    state.completionTimers.forEach((id) => window.clearTimeout(id));
    state.completionTimers = [];
  }

  function renderSummary(summary) {
    els.summaryHeader.innerHTML = [
      metricTemplate("Mean Score", `${Math.round(summary.meanScore * 100)}%`, "Across all predicted abnormalities", "rgba(108,92,231,0.16)"),
      metricTemplate("Elevated Features", `${summary.elevatedFeatures}`, "Scores above 60% probability", "rgba(240,113,103,0.18)"),
      metricTemplate("Regions Found", `${summary.regionsFound}`, "Distinct suspicious zones localized", "rgba(79,142,247,0.18)"),
      metricTemplate("Status", summary.modelStatus, state.isLoading ? "Inference pipeline active" : "System idle and ready", "rgba(0,180,216,0.2)")
    ].join("");
  }

  function renderModelInfo(modelInfo) {
    els.modelInfoList.innerHTML = Object.entries(modelInfo)
      .map(([key, value]) => {
        return `<li class="model-info-item"><span class="model-key">${key}</span><span class="model-val">${value}</span></li>`;
      })
      .join("");
  }

  function renderAbnormalities(predictions) {
    if (!predictions.length && !state.isLoading) {
      els.abnormalityList.innerHTML = `<div class="compare-placeholder" style="position:static; min-height:140px; border:1px dashed var(--border); border-radius:12px;">Run analysis to populate abnormalities</div>`;
      return;
    }

    if (state.isLoading) {
      const targetCount = getPredictionTargetCount();
      els.abnormalityList.innerHTML = Array.from({ length: targetCount })
        .map(() => `<div class="loading-skeleton skeleton-row"></div>`)
        .join("");
      return;
    }

    els.abnormalityList.innerHTML = predictions
      .map((prediction, idx) => {
        const severity = severityColor(prediction.score);
        const pct = Math.round(prediction.score * 100);
        return `
          <article class="abnormality-item surf-interactive" style="--severity-glow:${severity.glow}; animation-delay:${idx * 50}ms;">
            <div class="abn-head">
              <div class="abn-name">${prediction.label}</div>
              <div class="abn-score">${pct}%</div>
            </div>
            <div class="abn-track">
              <div class="abn-fill" style="--fill-from:${severity.from}; --fill-to:${severity.to}; width:${pct}%;"></div>
            </div>
            <div class="abn-meta">
              <div>${severity.label} significance</div>
              <div class="abn-status done"><span class="mini-dot"></span>Completed</div>
            </div>
          </article>
        `;
      })
      .join("");
  }

  function setGlobalAbnormalityCopy(predictions) {
    const btn = els.copyAllAbnormalitiesBtn;
    if (!btn) return;
    if (!predictions.length) {
      btn.disabled = true;
      btn.removeAttribute("data-copy");
      return;
    }
    const payloadLines = predictions.map((item) => {
      const pct = Math.round(item.score * 100);
      return `- ${item.label}: ${pct}%`;
    });
    const payload = `Abnormality Predictions (${predictions.length})\n${payloadLines.join("\n")}`;
    btn.setAttribute("data-copy", encodeURIComponent(payload));
    btn.disabled = false;
  }

  function renderHeatmaps(heatmaps) {
    if (!heatmaps.length && !state.isLoading) {
      els.heatmapGrid.innerHTML = `<div class="compare-placeholder" style="position:static; min-height:220px; border:1px dashed var(--border); border-radius:12px; grid-column:1 / -1;">Generated heatmaps appear here</div>`;
      return;
    }

    if (state.isLoading) {
      const targetCount = getHeatmapTargetCount();
      els.heatmapGrid.innerHTML = Array.from({ length: targetCount })
        .map(() => `<div class="loading-skeleton skeleton-thumb"></div>`)
        .join("");
      return;
    }

    els.heatmapGrid.innerHTML = heatmaps
      .map((heat, idx) => {
        const safeDownload = `heatmap-${idx + 1}.png`;
        return `
          <article class="heat-card surf-interactive ${idx === 0 ? "active" : ""}" data-heat-id="${idx}">
            <button class="heat-thumb surf-interactive ${idx === 0 ? "active" : ""}" data-heat-id="${idx}" type="button" aria-label="Open heatmap ${idx + 1}">
              <img src="${heat.src}" alt="${heat.label}" loading="lazy" />
              <span class="heat-tag">${heat.label}</span>
            </button>
            <div class="heat-actions">
              <button class="heat-open surf-interactive" type="button" data-heat-id="${idx}">Zoom</button>
              <a class="heat-download surf-interactive" href="${heat.src}" download="${safeDownload}" data-heat-id="${idx}">Download</a>
            </div>
          </article>
        `;
      })
      .join("");

    bindHeatmapSelection();
    bindSurfFx();
  }

  function renderProcessingFeed(predictions, heatmaps) {
    const predictionTarget = getPredictionTargetCount();
    const heatmapTarget = getHeatmapTargetCount();

    if (!state.isLoading) {
      const donePred = predictions.length;
      const doneHeat = heatmaps.length;
      els.processingFeed.innerHTML = `
        <div class="model-info-item"><span class="model-key">Abnormality Heads</span><span class="model-val">${donePred}/${predictionTarget} ready</span></div>
        <div class="model-info-item"><span class="model-key">Heatmap Renderers</span><span class="model-val">${doneHeat}/${heatmapTarget} ready</span></div>
        <div class="model-info-item"><span class="model-key">Pipeline</span><span class="model-val" style="color:var(--success)">Completed</span></div>
      `;
      return;
    }

    els.processingFeed.innerHTML = `
      <div class="model-info-item"><span class="model-key">Feature Heads</span><span class="model-val" id="feedHeadCount">0/${predictionTarget}</span></div>
      <div class="model-info-item"><span class="model-key">Heatmaps</span><span class="model-val" id="feedHeatCount">0/${heatmapTarget}</span></div>
      <div class="model-info-item"><span class="model-key">Inference</span><span class="model-val"><span class="dot-wave"><span></span><span></span><span></span></span></span></div>
    `;

    for (let i = 1; i <= predictionTarget; i += 1) {
      const t = window.setTimeout(() => {
        const head = document.getElementById("feedHeadCount");
        if (head) head.textContent = `${Math.min(i, predictionTarget)}/${predictionTarget}`;
      }, i * 240);
      state.completionTimers.push(t);
    }

    for (let i = 1; i <= heatmapTarget; i += 1) {
      const t = window.setTimeout(() => {
        const heat = document.getElementById("feedHeatCount");
        if (heat) heat.textContent = `${Math.min(i, heatmapTarget)}/${heatmapTarget}`;
      }, i * 260 + 180);
      state.completionTimers.push(t);
    }
  }

  function bindUploadDropzone() {
    ["dragenter", "dragover"].forEach((eventName) => {
      els.uploadDropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        els.uploadDropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      els.uploadDropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        els.uploadDropzone.classList.remove("dragover");
      });
    });

    els.uploadDropzone.addEventListener("drop", (event) => {
      const [file] = event.dataTransfer.files || [];
      if (file) {
        handleSelectedFile(file);
      }
    });

    els.uploadInput.addEventListener("change", (event) => {
      const [file] = event.target.files || [];
      if (file) {
        handleSelectedFile(file);
      }
    });
  }

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      els.uploadFileName.textContent = "Unsupported format";
      return;
    }
    state.selectedFile = file;
    els.uploadFileName.textContent = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
      state.sourceImage = e.target.result;
      els.sourcePreview.src = state.sourceImage;
      els.sourcePreview.style.display = "block";
      els.sourcePlaceholder.style.display = "none";
      els.runAnalysisBtn.disabled = false;
      els.heatmapPreview.removeAttribute("src");
      els.heatmapPreview.style.display = "none";
      els.heatPlaceholder.style.display = "grid";
      setResults({
        summary: state.config.summaryDefaults,
        predictions: [],
        heatmaps: [],
        sourceImage: state.sourceImage
      });
    };
    reader.readAsDataURL(file);
  }

  function bindComparisonToggle() {
    els.comparisonToggleGroup.addEventListener("click", (event) => {
      const btn = event.target.closest(".toggle-btn");
      if (!btn) return;
      const mode = btn.dataset.mode;
      state.comparisonMode = mode;
      [...els.comparisonToggleGroup.querySelectorAll(".toggle-btn")].forEach((el) => {
        const active = el === btn;
        el.classList.toggle("active", active);
        el.setAttribute("aria-selected", active ? "true" : "false");
      });
      els.comparisonBody.classList.toggle("single-mode", mode === "single");
    });
  }

  function bindHeatmapSelection() {
    [...els.heatmapGrid.querySelectorAll("[data-heat-id]")].forEach((el) => {
      if (el.dataset.boundHeat) return;
      el.dataset.boundHeat = "1";
      el.addEventListener("click", (event) => {
        const target = event.target.closest("[data-heat-id]");
        if (!target) return;
        const idx = Number(target.dataset.heatId);
        selectHeatmap(idx);

        if (event.target.closest(".heat-open, .heat-thumb")) {
          openHeatmapModal(idx);
        }
      });
    });
  }

  function selectHeatmap(idx) {
    [...els.heatmapGrid.querySelectorAll(".heat-thumb, .heat-card")].forEach((el) => el.classList.remove("active"));
    const selectedThumb = els.heatmapGrid.querySelector(`.heat-thumb[data-heat-id="${idx}"]`);
    const selectedCard = els.heatmapGrid.querySelector(`.heat-card[data-heat-id="${idx}"]`);
    if (selectedThumb) selectedThumb.classList.add("active");
    if (selectedCard) selectedCard.classList.add("active");

    const heat = state.heatmaps[idx];
    if (heat) {
      els.heatmapPreview.src = heat.src;
      els.heatmapPreview.style.display = "block";
      els.heatPlaceholder.style.display = "none";
    }
  }

  function applyZoomScale() {
    if (els.modalHeatmapImage) {
      els.modalHeatmapImage.style.transform = `scale(${modalState.scale})`;
    }
  }


  function updateModalNavigationControls() {
    const total = state.heatmaps.length;
    const idx = Number(modalState.activeHeatmap);

    if (els.prevHeatmapBtn) {
      els.prevHeatmapBtn.disabled = !(Number.isFinite(idx) && idx > 0);
    }
    if (els.nextHeatmapBtn) {
      els.nextHeatmapBtn.disabled = !(Number.isFinite(idx) && idx < total - 1);
    }
  }


  function populateCompareSelect() {
    if (!els.compareHeatmapSelect) return;
    const current = Number(modalState.activeHeatmap);
    const options = ['<option value="">Select heatmap</option>'];
    state.heatmaps.forEach((h, idx) => {
      if (idx === current) return;
      options.push(`<option value="${idx}">${h.label || `Heatmap ${idx + 1}`}</option>`);
    });
    els.compareHeatmapSelect.innerHTML = options.join("");

    if (modalState.compareHeatmap != null && modalState.compareHeatmap !== current) {
      els.compareHeatmapSelect.value = String(modalState.compareHeatmap);
    } else {
      modalState.compareHeatmap = null;
      els.compareHeatmapSelect.value = "";
    }
  }

  function renderComparePreview() {
    if (!els.comparePane || !els.compareHeatmapImage) return;
    const idx = Number(modalState.compareHeatmap);
    const valid = Number.isFinite(idx) && idx >= 0 && idx < state.heatmaps.length;
    const heat = valid ? state.heatmaps[idx] : null;

    if (heat && heat.src) {
      els.compareHeatmapImage.src = heat.src;
      els.comparePane.classList.add("has-compare");
    } else {
      els.compareHeatmapImage.removeAttribute("src");
      els.comparePane.classList.remove("has-compare");
    }
  }

  function setCompareMode(enabled) {
    modalState.compareMode = enabled;
    if (els.heatmapModal) {
      els.heatmapModal.classList.toggle("compare-mode", enabled);
    }
    if (els.compareToggleBtn) {
      els.compareToggleBtn.textContent = enabled ? "Exit Compare" : "Compare";
    }
    if (enabled) {
      populateCompareSelect();
      if (modalState.compareHeatmap == null && els.compareHeatmapSelect) {
        const fallback = els.compareHeatmapSelect.value;
        modalState.compareHeatmap = fallback !== "" ? Number(fallback) : null;
      }
      renderComparePreview();
    }
  }

  function openHeatmapModal(idx) {
    const heat = state.heatmaps[idx];
    if (!heat || !els.heatmapModal) return;

    modalState.activeHeatmap = idx;
    modalState.scale = 1;
    els.modalHeatmapImage.src = heat.src;
    if (modalState.compareHeatmap === idx) {
      modalState.compareHeatmap = null;
    }
    els.modalTitle.textContent = `${heat.label} - Zoom View`;
    els.modalDownloadBtn.href = heat.src;
    els.modalDownloadBtn.download = `heatmap-${idx + 1}.png`;
    applyZoomScale();
    updateModalNavigationControls();
    if (modalState.compareMode) {
      populateCompareSelect();
      renderComparePreview();
    }
    els.heatmapModal.classList.add("open");
    els.heatmapModal.setAttribute("aria-hidden", "false");
  }

  function closeHeatmapModal() {
    if (!els.heatmapModal) return;
    els.heatmapModal.classList.remove("open");
    els.heatmapModal.setAttribute("aria-hidden", "true");
  }

  function navigateHeatmapModal(step) {
    if (!state.heatmaps.length) return;
    const current = Number(modalState.activeHeatmap);
    if (!Number.isFinite(current)) return;
    const nextIdx = current + step;
    if (nextIdx < 0 || nextIdx >= state.heatmaps.length) return;
    openHeatmapModal(nextIdx);
  }

  function mockInferenceResults(config) {
    const fallbackCount = Number(config.abnormalityCount) || 8;
    const schema = Array.isArray(config.abnormalitySchema) && config.abnormalitySchema.length
      ? config.abnormalitySchema
      : Array.from({ length: fallbackCount }).map((_, idx) => `Abnormality ${idx + 1}`);

    const predictions = schema.map((label) => {
      const base = 0.18 + Math.random() * 0.72;
      return {
        label,
        score: Number(base.toFixed(2))
      };
    });

    const elevated = predictions.filter((p) => p.score >= 0.6).length;
    const mean = predictions.reduce((acc, item) => acc + item.score, 0) / predictions.length;
    const regions = Math.max(1, Math.round((elevated + mean * 10) / 2));

    const source = state.sourceImage;
    const desiredHeatmaps = Number(config.heatmapSlots) || predictions.length;
    const heatmaps = Array.from({ length: desiredHeatmaps }).map((_, i) => {
      return {
        label: `Map ${i + 1}`,
        src: source || createSyntheticHeatmap(i)
      };
    });

    return {
      summary: {
        meanScore: mean,
        elevatedFeatures: elevated,
        regionsFound: regions,
        modelStatus: "Ready"
      },
      predictions,
      heatmaps,
      sourceImage: source
    };
  }

  function createSyntheticHeatmap(seed) {
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext("2d");
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, "#0f172a");
    gradient.addColorStop(0.4, "#6c5ce7");
    gradient.addColorStop(0.7, "#4f8ef7");
    gradient.addColorStop(1, "#f97316");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < 7; i += 1) {
      const x = ((seed + 1) * 41 + i * 33) % canvas.width;
      const y = ((seed + 2) * 29 + i * 21) % canvas.height;
      const r = 20 + ((seed + i) % 6) * 14;
      const halo = ctx.createRadialGradient(x, y, 2, x, y, r);
      halo.addColorStop(0, "rgba(255,255,255,0.85)");
      halo.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    return canvas.toDataURL("image/png");
  }

  function setLoadingState(isLoading) {
    state.isLoading = isLoading;
    clearCompletionTimers();

    if (isLoading) {
      els.globalStatus.classList.add("loading");
      els.statusText.innerHTML = 'Status: Processing <span class="dot-wave" aria-hidden="true"><span></span><span></span><span></span></span>';
      els.runAnalysisBtn.classList.add("processing");
      els.runBtnText.innerHTML = 'Processing <span class="dot-wave" aria-hidden="true"><span></span><span></span><span></span></span>';
      els.runAnalysisBtn.disabled = true;
      renderSummary({
        meanScore: 0,
        elevatedFeatures: 0,
        regionsFound: 0,
        modelStatus: "Processing"
      });
      renderProcessingFeed([], []);
      renderAbnormalities([]);
      renderHeatmaps([]);
      setGlobalAbnormalityCopy([]);
      return;
    }

    els.globalStatus.classList.remove("loading");
    els.statusText.textContent = "Status: Ready";
    els.runAnalysisBtn.classList.remove("processing");
    els.runBtnText.textContent = "Run Analysis";
    els.runAnalysisBtn.disabled = !state.sourceImage;
  }

  function setResults({ summary, predictions, heatmaps, sourceImage }) {
    if (sourceImage) {
      state.sourceImage = sourceImage;
      els.sourcePreview.src = sourceImage;
      els.sourcePreview.style.display = "block";
      els.sourcePlaceholder.style.display = "none";
    }

    state.predictions = predictions || [];
    state.heatmaps = heatmaps || [];

    if (state.heatmaps[0]) {
      els.heatmapPreview.src = state.heatmaps[0].src;
      els.heatmapPreview.style.display = "block";
      els.heatPlaceholder.style.display = "none";
    } else {
      els.heatmapPreview.removeAttribute("src");
      els.heatmapPreview.style.display = "none";
      els.heatPlaceholder.style.display = "grid";
    }

    const elevatedCount = (state.predictions || []).filter((p) => p.score >= 0.6).length;
    els.predictionCounter.textContent = `${elevatedCount} / ${state.predictions.length || 0} elevated`;

    renderSummary(summary);
    renderAbnormalities(state.predictions);
    renderHeatmaps(state.heatmaps);
    renderProcessingFeed(state.predictions, state.heatmaps);
    setGlobalAbnormalityCopy(state.predictions);
  }

  function renderModelPage(modalityKey) {
    const selected = window.MODEL_PAGE_CONFIG[modalityKey] || window.MODEL_PAGE_CONFIG["brain-mri"];
    state.config = selected;
    state.modality = selected.key;

    document.documentElement.style.setProperty("--accent-primary", selected.themeAccent.includes("var(") ? getComputedStyle(document.documentElement).getPropertyValue(selected.themeAccent.replace("var(", "").replace(")", "")).trim() : selected.themeAccent);

    els.pageTitle.textContent = selected.title;
    els.pageSubtitle.textContent = selected.subtitle;
    els.brandIcon.textContent = selected.icon;
    els.serviceLabel.textContent = `${selected.serviceLabel} Model`;
    els.todayStamp.textContent = formatDateStamp();

    els.abnormalityList.classList.toggle("bento", selected.layoutVariant === "bento");

    renderModelInfo(selected.modelInfo);
    const countHint = `${getPredictionTargetCount()} variable abnormalities`;
    const heatHint = `${getHeatmapTargetCount()} variable heatmaps`;
    const panelSubtitle = document.querySelector(".workspace-grid .panel-sub");
    if (panelSubtitle) panelSubtitle.textContent = `Scrollable confidence list for ${countHint}.`;
    const heatSubtitle = document.querySelectorAll(".workspace-grid .panel-sub")[1];
    if (heatSubtitle) heatSubtitle.textContent = `Dynamic gallery for ${heatHint} with quick-compare selection.`;
    setResults({
      summary: selected.summaryDefaults,
      predictions: [],
      heatmaps: [],
      sourceImage: state.sourceImage
    });
    setLoadingState(false);
  }


  function buildResultsFromApi(apiResponse) {
    const schema = Array.isArray(state.config?.abnormalitySchema) && state.config.abnormalitySchema.length
      ? state.config.abnormalitySchema
      : ["Prediction"];

    const backendScores = Array.isArray(apiResponse?.head_scores) ? apiResponse.head_scores : [];
    let predictions = [];

    if (backendScores.length) {
      predictions = backendScores.map((item, idx) => {
        const label = String(item?.label || schema[idx] || `Abnormality ${idx + 1}`);
        const score = Math.max(0, Math.min(1, Number(item?.score ?? 0)));
        return {
          label,
          score: Number(score.toFixed(4))
        };
      });
    } else {
      const confidence = Number(apiResponse?.confidence ?? 0.0);
      const normalizedConfidence = Math.max(0, Math.min(1, confidence));
      const baseLabel = String(apiResponse?.prediction || schema[0] || "Prediction");
      predictions = [{
        label: baseLabel,
        score: Number(normalizedConfidence.toFixed(4))
      }];
    }

    const elevated = predictions.filter((p) => p.score >= 0.6).length;
    const mean = predictions.reduce((acc, item) => acc + item.score, 0) / Math.max(predictions.length, 1);
    const regions = Math.max(1, Math.round((elevated + mean * 10) / 2));

    const backendHeatmaps = Array.isArray(apiResponse?.heatmaps) ? apiResponse.heatmaps : [];
    let heatmaps = [];

    if (backendHeatmaps.length) {
      heatmaps = backendHeatmaps
        .filter((item) => item && item.src)
        .map((item, idx) => ({
          label: String(item.label || `Map ${idx + 1}`),
          src: String(item.src)
        }));
    }

    if (!heatmaps.length && state.sourceImage) {
      heatmaps = [{
        label: "No backend heatmap",
        src: state.sourceImage
      }];
    }

    return {
      summary: {
        meanScore: Number.isFinite(Number(apiResponse?.mean_s)) ? Number(apiResponse.mean_s) / 100 : mean,
        elevatedFeatures: Number.isFinite(Number(apiResponse?.high_n)) ? Number(apiResponse.high_n) : elevated,
        regionsFound: Number.isFinite(Number(apiResponse?.n_regions)) ? Number(apiResponse.n_regions) : regions,
        modelStatus: "Ready"
      },
      predictions,
      heatmaps,
      sourceImage: state.sourceImage,
      note: apiResponse?.note || ""
    };
  }

  async function inferWithBackend() {
    const endpoint = API_ENDPOINTS[state.modality] || "/predict/brain";
    const formData = new FormData();
    formData.append("file", state.selectedFile);

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `Request failed with status ${response.status}`);
    }

    return payload;
  }

  async function runAnalysis() {
    if (state.isLoading || !state.config || !state.selectedFile) return;
    setLoadingState(true);

    try {
      const apiResponse = await inferWithBackend();
      const results = buildResultsFromApi(apiResponse);
      setLoadingState(false);
      setResults(results);
      const backend = String(apiResponse?.backend || "unknown");
      const hash = String(apiResponse?.input_sha12 || "-");
      const note = results.note ? `; ${results.note}` : "";
      els.statusText.textContent = `Status: Ready [${backend}] hash:${hash}${note}`;
    } catch (error) {
      setLoadingState(false);
      els.statusText.textContent = `Status: Error (${error.message})`;
    }
  }

  function createCursorClickBurst(x, y) {
    const ring = document.createElement("div");
    ring.className = "cursor-click";
    ring.style.left = `${x}px`;
    ring.style.top = `${y}px`;
    document.body.appendChild(ring);
    window.setTimeout(() => ring.remove(), 560);

    for (let i = 0; i < 7; i += 1) {
      const shard = document.createElement("div");
      const angle = (Math.PI * 2 * i) / 7;
      const dist = 16 + Math.random() * 18;
      shard.style.cssText = `
        position: fixed;
        width: 4px;
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(130deg, var(--accent-primary), var(--accent-secondary));
        left: ${x}px;
        top: ${y}px;
        pointer-events: none;
        z-index: 9999;
        transform: translate(-50%, -50%);
        opacity: 1;
        transition: transform 420ms ease, opacity 420ms ease;
      `;
      document.body.appendChild(shard);
      requestAnimationFrame(() => {
        shard.style.transform = `translate(calc(-50% + ${Math.cos(angle) * dist}px), calc(-50% + ${Math.sin(angle) * dist}px))`;
        shard.style.opacity = "0";
      });
      window.setTimeout(() => shard.remove(), 440);
    }
  }

  function bindCursorFx() {
    if (!window.matchMedia("(pointer:fine)").matches || !els.cursorDot || !els.cursorRing) return;

    let ringX = window.innerWidth / 2;
    let ringY = window.innerHeight / 2;
    let mouseX = ringX;
    let mouseY = ringY;

    els.cursorDot.style.opacity = "1";
    els.cursorRing.style.opacity = "1";

    const animate = () => {
      ringX += (mouseX - ringX) * 0.2;
      ringY += (mouseY - ringY) * 0.2;
      els.cursorRing.style.left = `${ringX}px`;
      els.cursorRing.style.top = `${ringY}px`;
      state.particleAnimationFrame = window.requestAnimationFrame(animate);
    };
    animate();

    document.addEventListener("mousemove", (event) => {
      mouseX = event.clientX;
      mouseY = event.clientY;
      els.cursorDot.style.left = `${mouseX}px`;
      els.cursorDot.style.top = `${mouseY}px`;
    });

    document.addEventListener("mousedown", (event) => {
      createCursorClickBurst(event.clientX, event.clientY);
    });

    document.addEventListener("mouseover", (event) => {
      const interactive = event.target.closest("button, a, input, label, .surf-interactive");
      if (interactive) els.cursorRing.classList.add("hovering");
    });
    document.addEventListener("mouseout", (event) => {
      const interactive = event.target.closest("button, a, input, label, .surf-interactive");
      if (interactive) els.cursorRing.classList.remove("hovering");
    });
  }

  function bindSurfFx() {
    const selector = ".surf-interactive, .action-btn, .toggle-btn, .upload-dropzone, .heat-thumb, .heat-open, .heat-download, .copy-btn, .page-chip, .model-info-item, .status-pill";
    document.querySelectorAll(selector).forEach((el) => {
      el.classList.add("surf-interactive");
      if (!el.dataset.surfBound) {
        el.dataset.surfBound = "1";
        el.addEventListener("mousemove", (event) => {
          const rect = el.getBoundingClientRect();
          const mx = ((event.clientX - rect.left) / rect.width) * 100;
          const my = ((event.clientY - rect.top) / rect.height) * 100;
          el.style.setProperty("--mx", `${mx}%`);
          el.style.setProperty("--my", `${my}%`);
        });
      }
    });
  }

  function bindCopyButtons() {
    document.addEventListener("click", async (event) => {
      const btn = event.target.closest(".copy-btn");
      if (!btn) return;
      const encoded = btn.getAttribute("data-copy");
      if (!encoded) return;
      const text = decodeURIComponent(encoded);
      try {
        await navigator.clipboard.writeText(text);
        const original = btn.textContent;
        btn.textContent = "✓";
        btn.classList.add("copied");
        window.setTimeout(() => {
          btn.textContent = original || "⧉";
          btn.classList.remove("copied");
        }, 1100);
      } catch (_err) {
        btn.textContent = "!";
        window.setTimeout(() => {
          btn.textContent = "⧉";
        }, 1000);
      }
    });
  }

  function bindHeatmapModal() {
    if (!els.heatmapModal) return;

    els.heatmapModal.addEventListener("click", (event) => {
      if (event.target.closest("[data-close-modal='true']")) closeHeatmapModal();
    });
    els.closeModalBtn?.addEventListener("click", closeHeatmapModal);
    els.prevHeatmapBtn?.addEventListener("click", () => navigateHeatmapModal(-1));
    els.nextHeatmapBtn?.addEventListener("click", () => navigateHeatmapModal(1));
    els.compareToggleBtn?.addEventListener("click", () => {
      setCompareMode(!modalState.compareMode);
    });
    els.compareHeatmapSelect?.addEventListener("change", (event) => {
      const value = event.target.value;
      modalState.compareHeatmap = value === "" ? null : Number(value);
      renderComparePreview();
    });
    els.zoomInBtn?.addEventListener("click", () => {
      modalState.scale = Math.min(4, Number((modalState.scale + 0.2).toFixed(2)));
      applyZoomScale();
    });
    els.zoomOutBtn?.addEventListener("click", () => {
      modalState.scale = Math.max(0.6, Number((modalState.scale - 0.2).toFixed(2)));
      applyZoomScale();
    });
    els.zoomStage?.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 0.12 : -0.12;
      modalState.scale = Math.min(4, Math.max(0.6, Number((modalState.scale + delta).toFixed(2))));
      applyZoomScale();
    }, { passive: false });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeHeatmapModal();
        return;
      }
      if (!els.heatmapModal?.classList.contains("open")) return;
      if (event.key === "ArrowLeft") {
        navigateHeatmapModal(-1);
      } else if (event.key === "ArrowRight") {
        navigateHeatmapModal(1);
      }
    });
  }

  function bindParticleBackground() {
    const canvas = els.particleCanvas;
    if (!canvas || !canvas.getContext || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const ctx = canvas.getContext("2d");
    const dots = [];
    const count = 72;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    for (let i = 0; i < count; i += 1) {
      dots.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        r: 1 + Math.random() * 2.4,
        vx: -0.4 + Math.random() * 0.8,
        vy: -0.3 + Math.random() * 0.7,
        hue: [255, 210, 190][i % 3]
      });
    }

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const dot of dots) {
        dot.x += dot.vx;
        dot.y += dot.vy;
        if (dot.x < -10) dot.x = canvas.width + 10;
        if (dot.x > canvas.width + 10) dot.x = -10;
        if (dot.y < -10) dot.y = canvas.height + 10;
        if (dot.y > canvas.height + 10) dot.y = -10;

        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.r, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${dot.hue}, 90%, 62%, 0.34)`;
        ctx.fill();
      }

      for (let i = 0; i < dots.length; i += 1) {
        for (let j = i + 1; j < dots.length; j += 1) {
          const dx = dots[i].x - dots[j].x;
          const dy = dots[i].y - dots[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.strokeStyle = `rgba(108, 92, 231, ${0.12 - dist / 1200})`;
            ctx.lineWidth = 0.7;
            ctx.beginPath();
            ctx.moveTo(dots[i].x, dots[i].y);
            ctx.lineTo(dots[j].x, dots[j].y);
            ctx.stroke();
          }
        }
      }

      window.requestAnimationFrame(tick);
    };

    tick();
  }

  function init() {
    bindUploadDropzone();
    bindComparisonToggle();
    bindParticleBackground();
    bindCursorFx();
    bindCopyButtons();
    bindHeatmapModal();
    renderModelPage(state.modality);
    bindSurfFx();

    els.runAnalysisBtn.addEventListener("click", runAnalysis);

    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "r") {
        event.preventDefault();
        runAnalysis();
      }
    });
  }

  window.renderModelPage = renderModelPage;
  window.setLoadingState = setLoadingState;
  window.setResults = setResults;
  window.bindUploadDropzone = bindUploadDropzone;
  window.bindComparisonToggle = bindComparisonToggle;
  window.bindHeatmapSelection = bindHeatmapSelection;

  init();
})();
