const WINDOW    = 300;
const DRAIN_HZ  = 60;

const c3Data = new Array(WINDOW).fill(0);
const c4Data = new Array(WINDOW).fill(0);

const queue = [];

let rafId       = null;
let rafLastTs   = 0;
let pendingFrac = 0;

let activeModel = null;
let es          = null;
let correct     = 0;
let wrong       = 0;

const CHART_THEMES = {
  dark:  { c3: "#4f8ef7", c4: "#3de08a", grid: "rgba(255,255,255,0.04)", ticks: "#475569" },
  light: { c3: "#2f6fe0", c4: "#0f9d58", grid: "rgba(26,34,51,0.06)",    ticks: "#94a3b8" },
};
const chartTheme = () => CHART_THEMES[document.documentElement.dataset.theme] || CHART_THEMES.dark;

function makeChart(id, lineKey) {
  const ctx = document.getElementById(id).getContext("2d");
  const t = chartTheme();
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: new Array(WINDOW).fill(""),
      datasets: [{
        data: id === "chartC3" ? c3Data : c4Data,
        borderColor: t[lineKey],
        borderWidth: 1,
        pointRadius: 0,
        tension: 0,
        fill: false,
      }],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: {
          display: true,
          grid: { color: t.grid },
          ticks: {
            color: t.ticks,
            font: { size: 9 },
            maxTicksLimit: 4,
            callback: v => v.toFixed(1),
          },
        },
      },
      elements: { line: { capBezierPoints: false } },
    },
  });
}

const chartC3 = makeChart("chartC3", "c3");
const chartC4 = makeChart("chartC4", "c4");

window.applyChartTheme = () => {
  const t = chartTheme();
  for (const [chart, key] of [[chartC3, "c3"], [chartC4, "c4"]]) {
    chart.data.datasets[0].borderColor = t[key];
    chart.options.scales.y.grid.color = t.grid;
    chart.options.scales.y.ticks.color = t.ticks;
    chart.update("none");
  }
};

function startRaf() {
  if (rafId) return;
  rafLastTs   = performance.now();
  pendingFrac = 0;

  function tick(ts) {
    const dt = ts - rafLastTs;
    rafLastTs = ts;

    pendingFrac += dt * DRAIN_HZ / 1000;
    const drain = Math.min(Math.floor(pendingFrac), queue.length, 8);
    pendingFrac -= drain;

    if (drain > 0) {
      for (let i = 0; i < drain; i++) {
        const s = queue.shift();
        c3Data.shift(); c3Data.push(s.c3);
        c4Data.shift(); c4Data.push(s.c4);
      }
      chartC3.update("none");
      chartC4.update("none");
    }

    rafId = requestAnimationFrame(tick);
  }

  rafId = requestAnimationFrame(tick);
}

function stopRaf() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}

fetch("/models")
  .then(r => r.json())
  .then(names => {
    const grid = document.getElementById("modelGrid");
    names.forEach(name => {
      const btn = document.createElement("button");
      btn.className = "model-btn";
      btn.textContent = name.replace(/_/g, " ").toUpperCase();
      btn.dataset.model = name;
      btn.onclick = () => selectModel(name);
      grid.appendChild(btn);
    });
    if (names.length > 0) selectModel(names[0]);
  });

function selectModel(name) {
  activeModel = name;
  document.querySelectorAll(".model-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.model === name));
  setStatus(`Model: ${name.replace(/_/g, " ").toUpperCase()} — press Start.`, false);
}

function startStream() {
  if (!activeModel) return;
  if (es) es.close();

  correct = 0;
  wrong   = 0;
  queue.length = 0;
  updateStats(null);
  setStatus("Streaming …", true);
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled  = false;
  _setSlidersDisabled(true);

  startRaf();

  es = new EventSource(`/stream?model=${activeModel}`);

  es.onmessage = (e) => {
    const d = JSON.parse(e.data);

    if (d.done) {
      const finalAcc = (correct + wrong) > 0
        ? (correct / (correct + wrong) * 100).toFixed(1) : "—";
      stopStream();
      if (window.sim) window.sim.reset();
      setStatus(`Done — ${activeModel.replace(/_/g," ").toUpperCase()} accuracy: ${finalAcc}%`, false);
      return;
    }

    for (let i = 0; i < d.eeg_c3.length; i++) {
      queue.push({ c3: d.eeg_c3[i], c4: d.eeg_c4[i] });
    }

    const predIsLeft = d.prediction  === 0;
    const trueIsLeft = d.true_label  === 0;

    const trueEl = document.getElementById("trueLabel");
    trueEl.textContent = trueIsLeft ? "LEFT" : "RIGHT";
    trueEl.className   = `true-label ${trueIsLeft ? "left" : "right"}`;

    const predEl = document.getElementById("predLabel");
    predEl.textContent = predIsLeft ? "LEFT" : "RIGHT";
    if (d.correct) {
      predEl.className = `pred-label ${predIsLeft ? "correct-left" : "correct-right"}`;
    } else {
      predEl.className = "pred-label wrong";
    }

    const confPct = Math.round(d.confidence * 100);
    const bar = document.getElementById("confBar");
    bar.style.width      = `${confPct}%`;
    bar.style.background = predIsLeft ? "var(--accent)" : "var(--green)";
    document.getElementById("confText").textContent = `Confidence: ${confPct}%`;

    if (window.sim) window.sim.predict(predIsLeft ? "left" : "right");

    if (d.correct) correct++; else wrong++;
    updateStats(d);
  };

  es.onerror = () => {
    setStatus("Connection error — is the server running?", false);
    stopStream();
  };
}

function stopStream() {
  if (es) { es.close(); es = null; }
  stopRaf();
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled  = true;
  document.getElementById("statusDot").classList.remove("live");
  _setSlidersDisabled(false);
}

function updateStats(d) {
  const total = correct + wrong;
  const acc   = total > 0 ? correct / total : null;
  const accEl = document.getElementById("accVal");
  if (acc !== null) {
    accEl.textContent = `${(acc * 100).toFixed(1)}%`;
    accEl.className   = `stat-val ${acc >= 0.7 ? "good" : acc >= 0.5 ? "" : "bad"}`;
  } else {
    accEl.textContent = "—";
    accEl.className   = "stat-val";
  }
  document.getElementById("trialVal").textContent   = d ? `${d.trial}/${d.total}` : "—";
  document.getElementById("correctVal").textContent = correct || "—";
  document.getElementById("wrongVal").textContent   = wrong   || "—";
}

function setStatus(msg, live) {
  document.getElementById("statusText").textContent = msg;
  document.getElementById("statusDot").classList.toggle("live", live);
}

let animating = false;

function onHandSlider() {
  const l = document.getElementById("leftSlider").value / 100;
  const r = document.getElementById("rightSlider").value / 100;
  document.getElementById("leftVal").textContent  = Math.round(l * 100) + "%";
  document.getElementById("rightVal").textContent = Math.round(r * 100) + "%";

  if (animating) _setAnimate(false);
  if (window.sim) window.sim.setBlend(l, r);
}

function toggleAnimate() {
  _setAnimate(!animating);
}

function _setAnimate(on) {
  animating = on;
  const btn = document.getElementById("animBtn");
  btn.textContent = on ? "■ Stop" : "▶ Animate";
  btn.className   = on ? "btn btn-danger" : "btn btn-primary";
  if (window.sim) window.sim.oscillate(on);
}

function _setSlidersDisabled(disabled) {
  document.getElementById("leftSlider").disabled  = disabled;
  document.getElementById("rightSlider").disabled = disabled;
  document.getElementById("animBtn").disabled     = disabled;
  if (disabled) _setAnimate(false);
}
