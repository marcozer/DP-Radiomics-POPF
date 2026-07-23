/* global window, document, fetch, FormData, XMLHttpRequest */

function byId(id) {
  return document.getElementById(id);
}

function text(el, value) {
  if (!el) return;
  el.textContent = value;
}

function fmtProb(x) {
  if (x === null || x === undefined) return null;
  const v = Number(x);
  if (!Number.isFinite(v)) return String(x);
  return `${(v * 100).toFixed(1)}% (${v.toFixed(3)})`;
}

function riskBadgeClass(group) {
  const g = String(group || '').toLowerCase();
  if (g.startsWith('low')) return 'low';
  if (g.startsWith('high')) return 'high';
  if (g.startsWith('inter')) return 'mid';
  return '';
}

async function initCaseSwitcher(patientId) {
  try {
    const res = await fetch('/api/cases');
    if (!res.ok) return;
    const data = await res.json();
    const cases = Array.isArray(data.cases) ? data.cases : [];
    if (!cases.length) return;

    const slider = byId('caseSlider');
    const label = byId('caseSliderLabel');
    const select = byId('caseSelect');
    if (!slider || !label || !select) return;

    select.innerHTML = '';
    cases.forEach((c) => {
      const opt = document.createElement('option');
      opt.value = c.patient_id;
      const statusBits = [];
      if (c.ct_present) statusBits.push('CT');
      if (c.seg_present) statusBits.push('Seg');
      if (c.has_coordinate) statusBits.push('Head');
      if (c.has_prediction) statusBits.push('Pred');
      const status = statusBits.length ? ` [${statusBits.join('+')}]` : '';
      opt.textContent = `${c.patient_id}${status}`;
      select.appendChild(opt);
    });

    const currentIndex = Math.max(0, cases.findIndex((c) => c.patient_id === patientId));
    slider.min = 0;
    slider.max = Math.max(0, cases.length - 1);
    slider.value = String(currentIndex);
    select.value = patientId;

    function renderIndex(i) {
      const c = cases[i];
      if (!c) return;
      const when = c.updated_at || c.created_at || '';
      const extra = c.scanner_id ? ` · ${c.scanner_id}` : '';
      label.textContent = `${c.patient_id}${extra}${when ? ` · ${when}` : ''}`;
    }

    renderIndex(currentIndex);

    slider.addEventListener('input', () => {
      const idx = Number(slider.value);
      renderIndex(idx);
      const c = cases[idx];
      if (c) select.value = c.patient_id;
    });

    slider.addEventListener('change', () => {
      const idx = Number(slider.value);
      const c = cases[idx];
      if (c && c.patient_id && c.patient_id !== patientId) {
        window.location.href = `/cases/${encodeURIComponent(c.patient_id)}`;
      }
    });

    select.addEventListener('change', () => {
      const id = select.value;
      if (id && id !== patientId) {
        window.location.href = `/cases/${encodeURIComponent(id)}`;
      }
    });
  } catch (e) {
    // ignore
  }
}

function setProgress(prefix, job, logData) {
  const progressEl = byId(`${prefix}Progress`);
  const progressLabelEl = byId(`${prefix}ProgressLabel`);
  const detailsEl = byId(`${prefix}LogDetails`);
  const logEl = byId(`${prefix}Log`);

  const pct = (logData && typeof logData.progress_percent === 'number') ? logData.progress_percent : null;
  const stage = (logData && (logData.phase_detail || logData.stage_hint)) ? (logData.phase_detail || logData.stage_hint) : null;

  const running = job && (job.status === 'running' || job.status === 'queued');

  if (progressEl) {
    if (running) {
      progressEl.style.display = 'block';
      if (pct === null) {
        progressEl.removeAttribute('value');
      } else {
        progressEl.value = pct;
        progressEl.max = 100;
      }
    } else {
      progressEl.style.display = 'none';
    }
  }

  if (progressLabelEl) {
    if (running) {
      progressLabelEl.style.display = 'block';
      const pctText = pct === null ? '' : ` · ${pct}%`;
      progressLabelEl.textContent = stage ? `${stage}${pctText}` : (pct === null ? 'working…' : `${pct}%`);
    } else {
      progressLabelEl.style.display = 'none';
    }
  }

  if (detailsEl && logEl) {
    if (running || detailsEl.open) {
      detailsEl.style.display = 'block';
      logEl.textContent = (logData && logData.log_tail) ? logData.log_tail : '';
    } else {
      detailsEl.style.display = 'none';
    }
  }
}

async function fetchJobLog(jobId, tailBytes) {
  const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/log?tail_bytes=${tailBytes}`);
  if (!r.ok) return null;
  return r.json();
}

async function refreshPipelineStatus(patientId) {
  try {
    const r = await fetch(`/api/cases/${encodeURIComponent(patientId)}/status`);
    if (!r.ok) return { nextIntervalMs: 2000 };
    const data = await r.json();

    const preprocessEl = byId('preprocessStatus');
    const coordEl = byId('coordinateStatus');
    const predEl = byId('predictStatus');

    const resultRiskEl = byId('resultRisk');
    const resultGroupEl = byId('resultGroup');
    const resultCutpointsEl = byId('resultCutpoints');
    const downloadLinks = byId('downloadLinks');
    const downloadPred = byId('downloadPred');
    const downloadManifest = byId('downloadManifest');

    const ctReady = Boolean(data.ct_present && data.seg_present);
    const coord = data.saved_coordinate;

    const preprocessJob = data.latest_preprocess_job;
    if (preprocessJob) {
      const phase = preprocessJob.phase_detail ? ` · ${preprocessJob.phase_detail}` : '';
      text(preprocessEl, `${preprocessJob.status}${phase} (job ${String(preprocessJob.job_id).slice(0, 8)})`);
      const needLog = preprocessJob.status === 'running' || preprocessJob.status === 'queued' || (byId('preprocessLogDetails')?.open);
      const logData = needLog ? await fetchJobLog(preprocessJob.job_id, 20000) : null;
      setProgress('preprocess', preprocessJob, logData);
    } else if (ctReady) {
      text(preprocessEl, 'completed (CT + pancreas segmentation present)');
      setProgress('preprocess', null, null);
    } else {
      text(preprocessEl, 'waiting for DICOM upload (or CT/seg upload)');
      setProgress('preprocess', null, null);
    }

    if (coord && typeof coord === 'object') {
      const ts = coord.timestamp ? ` (saved ${coord.timestamp})` : '';
      text(coordEl, `saved${ts}`);
    } else {
      text(coordEl, 'not saved yet (use the embedded viewer, then click “Save Selection”)');
    }

    const computeBtn = byId('computeRiskButton');
    const computeHint = byId('computeRiskHint');
    if (computeBtn) {
      const canCompute = ctReady && coord && typeof coord === 'object';
      computeBtn.disabled = !canCompute;
      if (computeHint) {
        computeHint.textContent = canCompute ? 'Ready.' : 'Waiting for CT/seg + saved coordinate.';
      }
    }

    const predictJob = data.latest_predict_job;
    if (predictJob) {
      const phase = predictJob.phase_detail ? ` · ${predictJob.phase_detail}` : '';
      text(predEl, `${predictJob.status}${phase} (job ${String(predictJob.job_id).slice(0, 8)})`);
      const needLog = predictJob.status === 'running' || predictJob.status === 'queued' || (byId('predictLogDetails')?.open);
      const logData = needLog ? await fetchJobLog(predictJob.job_id, 20000) : null;
      setProgress('predict', predictJob, logData);
    } else {
      text(predEl, 'not started');
      setProgress('predict', null, null);
    }

    const pred = data.latest_prediction;
    if (pred && typeof pred === 'object') {
      const cal = pred.popf_risk_calibrated;
      const raw = pred.popf_risk_raw;
      const prob = (cal !== null && cal !== undefined) ? cal : raw;
      const src = (cal !== null && cal !== undefined) ? 'Reportable' : 'Raw';

      if (resultRiskEl) {
        const v = Number(prob);
        if (Number.isFinite(v)) {
          resultRiskEl.textContent = `${(v * 100).toFixed(1)}%`;
        } else {
          resultRiskEl.textContent = '-';
        }
      }

      if (resultGroupEl) {
        const group = pred.risk_group || '';
        resultGroupEl.textContent = group ? `Risk group: ${group}` : '';
        resultGroupEl.classList.remove('badge', 'low', 'mid', 'high');
        if (group) {
          resultGroupEl.classList.add('badge');
          const cls = riskBadgeClass(group);
          if (cls) resultGroupEl.classList.add(cls);
        }
      }

      if (resultCutpointsEl) {
        const lines = [];
        lines.push(`${src} POPF risk: ${fmtProb(prob) || '-'}`);
        if (pred.risk_threshold_low !== null && pred.risk_threshold_low !== undefined &&
            pred.risk_threshold_high !== null && pred.risk_threshold_high !== undefined) {
          const lo = Number(pred.risk_threshold_low);
          const hi = Number(pred.risk_threshold_high);
          if (Number.isFinite(lo) && Number.isFinite(hi)) {
            lines.push(`Exploratory cutpoints: rule-out < ${lo.toFixed(3)} · rule-in ≥ ${hi.toFixed(3)}`);
          }
        }
        resultCutpointsEl.textContent = lines.join('\n');
      }

      if (downloadLinks && predictJob && predictJob.status === 'completed') {
        downloadLinks.style.display = 'block';
        if (downloadPred) downloadPred.href = `/jobs/${encodeURIComponent(predictJob.job_id)}/files/popf_predictions.csv`;
        if (downloadManifest) downloadManifest.href = `/jobs/${encodeURIComponent(predictJob.job_id)}/files/manifest.json`;
      } else if (downloadLinks) {
        downloadLinks.style.display = 'none';
      }
    } else {
      if (resultRiskEl) resultRiskEl.textContent = '-';
      if (resultGroupEl) resultGroupEl.textContent = '';
      if (resultCutpointsEl) resultCutpointsEl.textContent = '';
      if (downloadLinks) downloadLinks.style.display = 'none';
    }

    const anyRunning =
      (preprocessJob && (preprocessJob.status === 'running' || preprocessJob.status === 'queued')) ||
      (predictJob && (predictJob.status === 'running' || predictJob.status === 'queued'));
    return { nextIntervalMs: anyRunning ? 1200 : 2200 };
  } catch (e) {
    return { nextIntervalMs: 2500 };
  }
}

function attachAjaxPost(form, { onDone } = {}) {
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const btn = form.querySelector('button[type="submit"]');
    const originalLabel = btn ? btn.textContent : null;
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Working…';
    }
    try {
      const res = await fetch(form.action, { method: 'POST', body: new FormData(form) });
      if (!res.ok) {
        const html = await res.text();
        document.open();
        document.write(html || '');
        document.close();
        return;
      }
      if (typeof onDone === 'function') onDone();
    } catch (e) {
      // ignore
    } finally {
      if (btn) {
        btn.disabled = false;
        if (originalLabel !== null) btn.textContent = originalLabel;
      }
    }
  });
}

function attachUploadForm(form, progressWrap) {
  form.addEventListener('submit', (ev) => {
    const fileInputs = Array.from(form.querySelectorAll('input[type="file"]'));
    const hasFile = fileInputs.some((i) => i.files && i.files.length);
    if (!hasFile) return;

    ev.preventDefault();

    const progressEl = progressWrap ? progressWrap.querySelector('progress') : null;
    const labelEl = progressWrap ? progressWrap.querySelector('[data-upload-label]') : null;
    if (progressWrap) progressWrap.classList.add('on');
    if (progressEl) {
      progressEl.value = 0;
      progressEl.max = 100;
    }
    if (labelEl) labelEl.textContent = 'Uploading…';

    const xhr = new XMLHttpRequest();
    xhr.open(form.method || 'POST', form.action, true);

    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.max(0, Math.min(100, Math.round((e.loaded / e.total) * 100)));
      if (progressEl) progressEl.value = pct;
      if (labelEl) labelEl.textContent = `Uploading… ${pct}%`;
    };

    xhr.onload = () => {
      const ok = xhr.status >= 200 && xhr.status < 400;
      if (ok) {
        const url = xhr.responseURL || window.location.href;
        window.location.href = url;
        return;
      }
      // Render server error page (keeps the helpful error list).
      try {
        document.open();
        document.write(xhr.responseText || '');
        document.close();
      } catch (e) {
        window.location.reload();
      }
    };

    xhr.onerror = () => {
      if (labelEl) labelEl.textContent = 'Upload failed (network error).';
    };

    xhr.send(new FormData(form));
  });
}

function hookSuggestedScanner() {
  try {
    const a = byId('useSuggestedScanner');
    const suggested = byId('suggestedScanner');
    const inp = byId('combatScannerInput');
    if (!a || !suggested || !inp) return;
    a.addEventListener('click', (ev) => {
      ev.preventDefault();
      inp.value = suggested.textContent.trim();
    });
  } catch (e) {
    // ignore
  }
}

async function main() {
  const cfg = window.RADPANC_CONFIG || {};
  const patientId = cfg.patientId;
  if (!patientId) return;

  await initCaseSwitcher(patientId);
  hookSuggestedScanner();

  // Upload progress bars
  const uploadWrap = byId('uploadProgressWrap');
  const dicomForm = byId('dicomUploadForm');
  const niftiForm = byId('niftiUploadForm');
  if (dicomForm) attachUploadForm(dicomForm, uploadWrap);
  if (niftiForm) attachUploadForm(niftiForm, uploadWrap);

  // Snappy POSTs (no full reload): preprocess rerun, ComBat save, compute risk
  const rerunForm = byId('rerunPreprocessForm');
  const combatForm = byId('combatForm');
  const predictForm = byId('predictForm');
  const refreshNow = () => refreshPipelineStatus(patientId);

  if (rerunForm) attachAjaxPost(rerunForm, { onDone: () => {
    const progressEl = byId('preprocessProgress');
    const progressLabelEl = byId('preprocessProgressLabel');
    if (progressEl) {
      progressEl.style.display = 'block';
      progressEl.removeAttribute('value');
    }
    if (progressLabelEl) {
      progressLabelEl.style.display = 'block';
      progressLabelEl.textContent = 'starting…';
    }
    const d = byId('preprocessLogDetails');
    if (d) d.open = true;
    refreshNow();
  }});
  if (combatForm) attachAjaxPost(combatForm, { onDone: refreshNow });
  if (predictForm) attachAjaxPost(predictForm, { onDone: () => {
    const pl = byId('pipeline');
    if (pl) pl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    const progressEl = byId('predictProgress');
    const progressLabelEl = byId('predictProgressLabel');
    if (progressEl) {
      progressEl.style.display = 'block';
      progressEl.removeAttribute('value');
    }
    if (progressLabelEl) {
      progressLabelEl.style.display = 'block';
      progressLabelEl.textContent = 'starting…';
    }
    const d = byId('predictLogDetails');
    if (d) d.open = true;
    refreshNow();
  }});

  // Poll loop with adaptive interval.
  async function loop() {
    const out = await refreshPipelineStatus(patientId);
    const nextMs = (out && out.nextIntervalMs) ? out.nextIntervalMs : 2000;
    window.setTimeout(loop, nextMs);
  }
  loop();

  // If navigated from a job redirect, auto-open log panels and scroll to pipeline.
  try {
    const qs = new URLSearchParams(window.location.search);
    const job = qs.get('job');
    if (job) {
      const pl = byId('pipeline');
      if (pl) pl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      const d1 = byId('preprocessLogDetails');
      const d2 = byId('predictLogDetails');
      if (d1) d1.open = true;
      if (d2) d2.open = true;
    }
  } catch (e) {
    // ignore
  }
}

document.addEventListener('DOMContentLoaded', main);
