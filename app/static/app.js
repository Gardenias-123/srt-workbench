const $ = (id) => document.getElementById(id);
const state = {
  demo: false,
  engineInfo: null,
  jobId: null,
  pollTimer: null,
  lastFortran: null,
  lastFortranJobId: null,
  lastJax: null,
  selectedVaa: null,
  busy: false,
};

const fmt = (x, d = 4) => {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(d);
};

function setStatus(text, kind = "") {
  const el = $("progressText");
  el.textContent = text;
  el.style.color = kind === "error" ? "var(--bad)" : kind === "ok" ? "var(--ok)" : "";
}

function setProgress(pct) {
  $("progressBar").style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function setBusy(on) {
  state.busy = on;
  document.querySelectorAll("button.primary").forEach((b) => {
    if (on) {
      if (!b.dataset.label) b.dataset.label = b.textContent;
      b.disabled = true;
      b.textContent = "运行中…";
    } else {
      b.disabled = false;
      if (b.dataset.label) b.textContent = b.dataset.label;
    }
  });
}

function num(id) {
  const v = parseFloat($(id).value);
  if (Number.isNaN(v)) throw new Error(`${id} 不是有效数字`);
  return v;
}

function numList(id) {
  const vals = $(id).value.split(",").map((s) => parseFloat(s.trim())).filter((v) => !Number.isNaN(v));
  if (!vals.length) throw new Error(`${id} 列表为空`);
  return vals;
}

function collectParams() {
  return {
    H: num("H"), cover: num("cover"), d_L: num("dL"), tree_diam: num("diam"),
    tree_type: parseInt($("treeType").value, 10), lad: parseInt($("lad").value, 10),
    tau_L: num("tauL"), ro_L: num("roL"), ro_soil: num("roSoil"),
    hot_spot: num("hotSpot"), ifren: 2, refind: 1.0, sk: 0.05,
    sza: num("sza"), saa: num("saa"), slope: num("slope"), aspect: num("aspect"),
    alpha_slope: num("alphaSlope"), beta_aspect: num("betaAspect"),
    coord_flag: 1, i0: num("i0"), nz: parseInt($("nz").value, 10),
    maxiter: parseInt($("maxiter").value, 10), epsiter: 1e-7,
    vza: numList("vza"), vaa: numList("vaa"),
  };
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function startJob(url, body, onDone) {
  if (state.busy) return;
  setBusy(true);
  setProgress(0);
  setStatus("提交任务…");
  $("logFull").textContent = "任务已提交，等待输出…";
  postJSON(url, body).then((data) => {
    state.jobId = data.job_id;
    pollJob(data.job_id, onDone);
  }).catch((err) => {
    setStatus(err.message, "error");
    setBusy(false);
  });
}

function pollJob(jobId, onDone) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    try {
      const snap = await getJSON(`/api/jobs/${jobId}`);
      setProgress(snap.progress);
      setStatus(`${snap.status === "running" ? "运行中" : snap.status} · ${snap.message || ""}`.trim());
      if (snap.log_tail && snap.log_tail.length) {
        $("logFull").textContent = snap.log_tail.join("\n");
        $("logFull").scrollTop = $("logFull").scrollHeight;
      }
      if (snap.status === "done") {
        const result = await getJSON(`/api/jobs/${jobId}/result`);
        setProgress(100);
        setStatus("完成", "ok");
        setBusy(false);
        onDone(result, jobId);
        return;
      }
      if (snap.status === "error" || snap.status === "cancelled") {
        setStatus(snap.error || snap.status, "error");
        setBusy(false);
        return;
      }
      pollJob(jobId, onDone);
    } catch (err) {
      setStatus(err.message, "error");
      setBusy(false);
    }
  }, 600);
}

// ---------------- Forward ----------------
function runForward() {
  try {
    const params = collectParams();
    if (state.demo) {
      startJob("/api/jax/forward", params, (result) => {
        state.lastJax = result;
        renderForward(result);
      });
      return;
    }
    startJob("/api/forward", params, (result, jobId) => {
      state.lastFortran = result;
      state.lastFortranJobId = jobId;
      renderForward(result);
    });
  } catch (err) { setStatus(err.message, "error"); }
}

function clearForwardCharts() {
  ["brfChart", "polarChart", "profileChart"].forEach((id) => {
    const el = $(id);
    if (el && el.data) Plotly.purge(el);
  });
  $("vaaSelect").innerHTML = "";
  $("energyTable").innerHTML = "";
  $("cardStatus").textContent = "运行中";
  $("cardEngine").textContent = "—";
  $("cardMean").textContent = "—";
  $("cardEnergy").textContent = "—";
  ["downloadCsv", "downloadJson"].forEach((id) => {
    $(id).classList.add("disabled");
    $(id).removeAttribute("href");
  });
}

function renderForward(result) {
  const angles = result.angles || [];
  if (!angles.length) { setStatus("结果为空", "error"); return; }
  $("cardStatus").textContent = "完成";
  $("cardEngine").textContent = result.engine === "fortran"
    ? "Fortran TopoSRT"
    : (result.engine === "jax" ? "JAX 可微模型" : result.engine);
  const brf = (result.columns && result.columns.brf_plane) || angles.map((a) => a.BRF);
  $("cardMean").textContent = fmt(brf.reduce((s, v) => s + v, 0) / brf.length, 4);
  const delta = result.energy && result.energy.bs_dir && result.energy.bs_dir.delta_percent;
  $("cardEnergy").textContent = delta === undefined ? "—" : `${fmt(delta, 2)}%`;

  const vaas = [...new Set(angles.map((a) => a.VAA))].sort((a, b) => a - b);
  state.selectedVaa = vaas[0];
  $("vaaSelect").innerHTML = vaas.map((v) => `<option value="${v}">${v}°</option>`).join("");
  $("vaaSelect").onchange = (e) => { state.selectedVaa = parseFloat(e.target.value); drawBrf(result); };
  drawBrf(result);
  drawPolar(result);
  drawProfile(result);
  drawEnergy(result);
  $("downloadCsv").classList.remove("disabled");
  $("downloadCsv").href = `/api/download/${state.jobId}/brf.csv`;
  $("downloadJson").classList.remove("disabled");
  $("downloadJson").href = `/api/download/${state.jobId}/result.json`;
  $("logFull").textContent = (result.log_tail || []).join("\n") || "无日志";
  $("compareStatus").textContent = state.lastJax ? "已有 JAX 结果，可在“对比”页查看。" : "Fortran 结果已就绪，可运行 JAX 对比。";
}

function drawBrf(result) {
  const angles = result.angles || [];
  const cols = result.columns || {};
  const idx = angles.map((a, i) => ({ a, i })).filter(({ a }) => a.VAA === state.selectedVaa);
  const x = idx.map(({ a }) => a.VZA);
  const plane = idx.map(({ i }) => cols.brf_plane ? cols.brf_plane[i] : angles[i].BRF);
  const slope = idx.map(({ i }) => cols.brf_slope ? cols.brf_slope[i] : angles[i].BRF);
  Plotly.react("brfChart", [
    { x, y: plane, type: "scatter", mode: "lines+markers", name: "平面系 BRF", line: { color: "#2563eb", width: 2.5 } },
    { x, y: slope, type: "scatter", mode: "lines+markers", name: "坡面系 BRF", line: { color: "#0ea5e9", width: 2, dash: "dot" } },
  ], chartLayout("VZA (°)", "BRF"), { responsive: true, displaylogo: false });
}

function chartLayout(xt, yt) {
  return {
    margin: { l: 54, r: 18, t: 24, b: 48 },
    xaxis: { title: xt, gridcolor: "#eef2f7" },
    yaxis: { title: yt, gridcolor: "#eef2f7" },
    legend: { orientation: "h", y: 1.12 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { family: "Segoe UI, Microsoft YaHei, sans-serif", size: 12 },
  };
}

function drawPolar(result) {
  const angles = result.angles || [];
  const cols = result.columns || {};
  const vzas = [...new Set(angles.map((a) => a.VZA))].sort((a, b) => a - b);
  const vaas = [...new Set(angles.map((a) => a.VAA))].sort((a, b) => a - b);
  const map = new Map();
  angles.forEach((a, i) => map.set(`${a.VZA}|${a.VAA}`, cols.brf_plane ? cols.brf_plane[i] : a.BRF));
  const z = vaas.map((vaa) => vzas.map((vza) => map.get(`${vza}|${vaa}`) ?? null));
  Plotly.react("polarChart", [{
    x: vzas, y: vaas, z, type: "heatmap", colorscale: "Viridis",
    colorbar: { title: "BRF", thickness: 12 },
    hovertemplate: "VZA=%{x}°<br>VAA=%{y}°<br>BRF=%{z:.4f}<extra></extra>",
  }], {
    margin: { l: 58, r: 20, t: 20, b: 48 },
    xaxis: { title: "VZA (°)" }, yaxis: { title: "VAA (°)" },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { family: "Segoe UI, Microsoft YaHei, sans-serif", size: 12 },
  }, { responsive: true, displaylogo: false });
}

function drawProfile(result) {
  const p = result.profile || {};
  if (!p.z || !p.z.length) { Plotly.purge("profileChart"); return; }
  Plotly.react("profileChart", [
    { x: p.u_dir, y: p.z, type: "scatter", mode: "lines", name: "U_dir", line: { color: "#2563eb", width: 2 } },
    { x: p.idir, y: p.z, type: "scatter", mode: "lines", name: "I_dir", line: { color: "#0ea5e9", width: 2 } },
    { x: p.p, y: p.z, type: "scatter", mode: "lines", name: "p(z)", line: { color: "#d97706", width: 2, dash: "dot" }, xaxis: "x2" },
  ], {
    margin: { l: 54, r: 54, t: 24, b: 48 },
    xaxis: { title: "U_dir / I_dir", gridcolor: "#eef2f7" },
    xaxis2: { title: "p(z)", overlaying: "x", side: "top", showgrid: false, range: [0, 1] },
    yaxis: { title: "z (m)", autorange: "reversed", gridcolor: "#eef2f7" },
    legend: { orientation: "h", y: 1.12 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { family: "Segoe UI, Microsoft YaHei, sans-serif", size: 12 },
  }, { responsive: true, displaylogo: false });
}

function drawEnergy(result) {
  const e = result.energy || {};
  if (!e.bs_dir && !e.bs_dif && !e.s) {
    $("energyTable").innerHTML = '<p class="hint">当前结果没有 Fortran 能量收支表（JAX 正演只输出 BRF 曲线）。</p>';
    return;
  }
  const rows = [];
  const add = (name, obj) => {
    if (!obj) return;
    rows.push({
      name,
      refl: obj.Refl ? obj.Refl[0] : null,
      trans: obj.Trans ? obj.Trans.reduce((s, v) => s + v, 0) : null,
      absor: obj.Absor ? obj.Absor.reduce((s, v) => s + v, 0) : null,
      delta: obj.delta_percent,
    });
  };
  add("BS_dir（直射，黑土）", e.bs_dir);
  add("BS_dif（漫射天空，黑土）", e.bs_dif);
  add("S（底部单位光源）", e.s);
  $("energyTable").innerHTML = `<table><thead><tr><th>子问题</th><th>Refl</th><th>Trans</th><th>Absor</th><th>delta</th></tr></thead><tbody>${
    rows.map((r) => `<tr><td>${r.name}</td><td>${fmt(r.refl)}</td><td>${fmt(r.trans)}</td><td>${fmt(r.absor)}</td><td>${r.delta === undefined ? "—" : fmt(r.delta, 2) + "%"}</td></tr>`).join("")
  }</tbody></table>`;
}

// ---------------- Compare ----------------
function runJaxCompare() {
  try {
    const params = collectParams();
    startJob("/api/jax/forward", params, (result) => {
      state.lastJax = result;
      renderCompare();
    });
  } catch (err) { $("compareStatus").textContent = err.message; }
}

function renderCompare() {
  const f = state.lastFortran;
  const j = state.lastJax;
  if (!f || !j) {
    $("compareStatus").textContent = "需要先运行 Fortran 正演，再运行 JAX 正演。";
    return;
  }
  const fa = f.angles || [];
  const ja = j.angles || [];
  const fc = (f.columns && f.columns.brf_plane) || [];
  const jc = (j.columns && j.columns.brf_plane) || [];
  const key = (a) => `${a.VZA}|${a.VAA}`;
  const jmap = new Map(ja.map((a, i) => [key(a), jc[i]]));
  const pairs = fa.map((a, i) => ({ a, f: fc[i], j: jmap.get(key(a)) })).filter((p) => p.j !== undefined);
  const rmse = Math.sqrt(pairs.reduce((s, p) => s + (p.f - p.j) ** 2, 0) / Math.max(pairs.length, 1));
  const bias = pairs.reduce((s, p) => s + (p.j - p.f), 0) / Math.max(pairs.length, 1);
  $("compareStatus").textContent = `匹配 ${pairs.length} 个角度 · RMSE=${fmt(rmse)} · 偏差=${fmt(bias)}`;
  const vaa = state.selectedVaa ?? (fa.length ? fa[0].VAA : 0);
  const fp = pairs.filter((p) => p.a.VAA === vaa).sort((a, b) => a.a.VZA - b.a.VZA);
  Plotly.react("compareChart", [
    { x: fp.map((p) => p.a.VZA), y: fp.map((p) => p.f), type: "scatter", mode: "lines+markers", name: "Fortran", line: { color: "#2563eb", width: 2.5 } },
    { x: fp.map((p) => p.a.VZA), y: fp.map((p) => p.j), type: "scatter", mode: "lines+markers", name: "JAX", line: { color: "#d97706", width: 2, dash: "dot" } },
  ], chartLayout("VZA (°)", "BRF"), { responsive: true, displaylogo: false });
  $("compareTable").innerHTML = `<table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>
    <tr><td>匹配角度数</td><td>${pairs.length}</td></tr>
    <tr><td>RMSE</td><td>${fmt(rmse)}</td></tr>
    <tr><td>平均偏差 (JAX - Fortran)</td><td>${fmt(bias)}</td></tr>
    <tr><td>Fortran 平均 BRF</td><td>${fmt(fc.reduce((s, v) => s + v, 0) / Math.max(fc.length, 1))}</td></tr>
    <tr><td>JAX 平均 BRF</td><td>${fmt(jc.reduce((s, v) => s + v, 0) / Math.max(jc.length, 1))}</td></tr>
  </tbody></table>`;
}

// ---------------- Inversion ----------------
function runInversion() {
  try {
    const params = collectParams();
    const source = $("invSource").value;
    if (source === "fortran" && !state.lastFortranJobId) {
      $("invStatus").textContent = "还没有 Fortran 正演结果，请先在“正演”页运行。";
      return;
    }
    const free = [...document.querySelectorAll("[data-free]:checked")].map((x) => x.dataset.free);
    if (!free.length) { $("invStatus").textContent = "至少选择一个反演参数。"; return; }
    const init = [
      params.H * 0.85, params.cover * 0.9, params.d_L * 0.8,
      Math.max(0, params.slope * 0.6), (params.aspect + 40) % 360,
    ];
    $("invStatus").textContent = "运行中…";
    startJob("/api/inversion", {
      params, source, job_id: state.lastFortranJobId, free, init,
      n_iter: parseInt($("invIter").value, 10), lr: parseFloat($("invLr").value),
      noise: parseFloat($("invNoise").value), sigma: 0.005,
    }, renderInversion);
  } catch (err) { $("invStatus").textContent = err.message; }
}

function renderInversion(result) {
  $("invStatus").textContent = `完成 · 来源 ${result.source} · RMSE=${fmt(result.rmse)} · 条件数=${fmt(result.condition_number, 1)}`;
  const loss = result.loss_history || [];
  Plotly.react("lossChart", [{
    x: loss.map((_, i) => i), y: loss, type: "scatter", mode: "lines",
    name: "loss", line: { color: "#2563eb", width: 2 },
  }], {
    margin: { l: 54, r: 18, t: 20, b: 44 },
    xaxis: { title: "iteration" }, yaxis: { title: "loss", type: "log" },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  }, { responsive: true, displaylogo: false });

  const obs = result.observed || [];
  const pred = result.predicted || [];
  Plotly.react("fitChart", [
    { x: obs.map((_, i) => i), y: obs, type: "scatter", mode: "markers", name: "observed", marker: { color: "#94a3b8", size: 5 } },
    { x: pred.map((_, i) => i), y: pred, type: "scatter", mode: "markers", name: "predicted", marker: { color: "#2563eb", size: 5 } },
  ], {
    margin: { l: 54, r: 18, t: 20, b: 44 },
    xaxis: { title: "observation index" }, yaxis: { title: "BRF" },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
  }, { responsive: true, displaylogo: false });

  const names = result.free_names || [];
  const truth = result.truth || {};
  const ret = result.free_retrieved || {};
  const std = result.std || [];
  const rows = names.map((n, i) => {
    const t = truth[n];
    const r = ret[n];
    const err = t ? ((r - t) / t) * 100 : null;
    return `<tr><td>${n}</td><td>${t === undefined ? "—" : fmt(t)}</td><td>${fmt(r)}</td><td>${fmt(std[i])}</td><td>${err === null ? "—" : fmt(err, 1) + "%"}</td></tr>`;
  }).join("");
  $("invTable").innerHTML = `<table><thead><tr><th>参数</th><th>真值</th><th>反演值</th><th>标准差</th><th>相对误差</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// ---------------- Sensitivity ----------------
function runSensitivity() {
  try {
    const params = collectParams();
    const source = $("sensSource").value;
    if (source === "fortran" && !state.lastFortranJobId) {
      $("sensStatus").textContent = "还没有 Fortran 正演结果，请先在“正演”页运行。";
      return;
    }
    $("sensStatus").textContent = "运行中…";
    startJob("/api/sensitivity", {
      params, source, job_id: state.lastFortranJobId, sigma: 0.005,
    }, renderSensitivity);
  } catch (err) { $("sensStatus").textContent = err.message; }
}

function renderSensitivity(result) {
  const names = result.names || [];
  const rel = result.relative_sensitivity || [];
  const nObs = rel.length ? rel[0].length : 0;
  const z = names.map((_, j) => rel.map((row) => row[j])); // 5 x n_obs
  Plotly.react("sensChart", [{
    x: Array.from({ length: nObs }, (_, i) => i), y: names, z, type: "heatmap",
    colorscale: "RdBu", zmid: 0, colorbar: { title: "rel. sens.", thickness: 12 },
    hovertemplate: "param=%{y}<br>obs=%{x}<br>value=%{z:.3g}<extra></extra>",
  }], {
    margin: { l: 70, r: 20, t: 20, b: 42 },
    xaxis: { title: "observation index" }, yaxis: { title: "" },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { family: "Segoe UI, Microsoft YaHei, sans-serif", size: 12 },
  }, { responsive: true, displaylogo: false });

  const corr = result.correlation || [];
  Plotly.react("corrChart", [{
    x: names, y: names, z: corr, type: "heatmap", colorscale: "RdBu", zmid: 0,
    colorbar: { title: "corr.", thickness: 12 },
    hovertemplate: "%{y} vs %{x}<br>%{z:.3f}<extra></extra>",
  }], {
    margin: { l: 70, r: 20, t: 20, b: 48 },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff",
    font: { family: "Segoe UI, Microsoft YaHei, sans-serif", size: 12 },
  }, { responsive: true, displaylogo: false });
  $("sensStatus").textContent = `完成 · 条件数=${fmt(result.condition_number, 1)}`;
  $("sensMeta").textContent = "相对敏感性 = ∂BRF/∂θ · θ / BRF；相关矩阵来自 Fisher 逆矩阵。条件数越大，参数越难分离。";
}

// ---------------- setup ----------------
function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      const pane = $(`tab-${btn.dataset.tab}`);
      if (pane) pane.classList.add("active");
    });
  });
}

function setupPresets() {
  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.dataset.preset;
      if (preset === "quick") { $("vza").value = "0,15,30,45,60"; $("vaa").value = "0,90,180,270"; }
      if (preset === "dense") { $("vza").value = "0,10,20,30,40,50,60,70"; $("vaa").value = "0,45,90,135,180,225,270,315"; }
      if (preset === "principal") { $("vza").value = "0,10,20,30,40,50,60"; $("vaa").value = "0,180"; }
    });
  });
}

async function checkHealth() {
  try {
    const h = await getJSON("/api/health");
    state.engineInfo = h;
    state.demo = h.fortran === false;
    $("enginePill").style.background = "rgba(16,185,129,.18)";
    if (state.demo) {
      $("enginePill").textContent = "JAX 引擎 · 在线演示";
      $("runBtn").textContent = "运行 JAX 正演";
      if ($("subtitle")) $("subtitle").textContent = "JAX 可微正演 · 梯度反演 · 敏感性（在线演示）";
      const tabBtn = document.querySelector('.tab[data-tab="compare"]');
      if (tabBtn) tabBtn.style.display = "none";
      const pane = $("tab-compare");
      if (pane) pane.style.display = "none";
      ["invSource", "sensSource"].forEach((id) => {
        const sel = $(id);
        if (!sel) return;
        [...sel.options].forEach((o) => {
          if (o.value === "fortran") { o.disabled = true; o.textContent = "Fortran（在线演示不可用）"; }
        });
        sel.value = "synthetic";
      });
      setStatus("在线演示：JAX 引擎（Fortran 已关闭）");
    } else {
      $("enginePill").textContent = `${h.engine} · ${String(h.exe_hash || "").slice(0, 8)}`;
    }
  } catch (err) {
    $("enginePill").textContent = "引擎不可用";
    $("enginePill").style.background = "rgba(220,38,38,.25)";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupPresets();
  checkHealth();
  $("runBtn").addEventListener("click", runForward);
  $("jaxCompareBtn").addEventListener("click", runJaxCompare);
  $("invBtn").addEventListener("click", runInversion);
  $("sensBtn").addEventListener("click", runSensitivity);
});
