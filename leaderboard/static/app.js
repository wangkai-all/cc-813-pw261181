/* 多源单调聚合 Top-K 分析查询网页逻辑（原生 JS，无依赖） */

const $ = (id) => document.getElementById(id);
let currentMeta = null;      // 当前数据集元信息
let stepSession = null;      // 步进会话 id
let sourceNames = [];

async function api(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

const fmt = (x) =>
  x === null || x === undefined
    ? "—"
    : Number.isFinite(x)
      ? Number(x).toFixed(6).replace(/0+$/, "").replace(/\.$/, "")
      : String(x);

// ---------------------------------------------------------------- 配置收集

function configBody(extra) {
  const weights = {};
  document.querySelectorAll("#weights input[data-source]").forEach((el) => {
    weights[el.dataset.source] = parseFloat(el.value) || 0;
  });
  const latencies = {};
  document.querySelectorAll("#latencies input[data-source]").forEach((el) => {
    latencies[el.dataset.source] = parseFloat(el.value) || 0;
  });
  return {
    name: $("dataset").value,
    k: parseInt($("k").value, 10),
    aggregator: $("aggregator").value,
    missing: $("missing").value,
    mode: $("mode").value,
    sa_policy: $("sa_policy").value,
    on_drift: $("on_drift").value,
    weights,
    latencies,
    ...(extra || {}),
  };
}

// ---------------------------------------------------------------- 数据集

async function loadDatasets() {
  const resp = await fetch("/api/datasets");
  const data = await resp.json();
  const sel = $("dataset");
  sel.innerHTML = "";
  data.datasets.forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = d.name;
    sel.appendChild(opt);
  });
  sel.value = "counterexample";
  await doLoad();
}

async function doLoad() {
  const body = configBody();
  const data = await api("/api/load", body);
  currentMeta = data;
  sourceNames = data.sources.map((s) => s.name);
  $("dataset-desc").textContent = data.description;
  $("k").value = data.k;
  $("aggregator").value = data.spec.aggregator;
  $("missing").value = data.spec.missing;
  renderWeightInputs(data);
  renderLatencyInputs(data);
  renderMutateSources(data);
  resetOutputs();
}

function renderWeightInputs(data) {
  const box = $("weights");
  box.innerHTML = "<label class='static-label'>权重 / 模拟延迟 →</label>";
  data.sources.forEach((s) => {
    const w = data.spec.weights[s.name] ?? 1;
    box.insertAdjacentHTML(
      "beforeend",
      `<label>${s.name} 权重
        <input type="number" step="0.1" data-source="${s.name}"
               class="w-input" value="${w}"></label>`
    );
  });
}

function renderLatencyInputs(data) {
  const box = $("latencies");
  box.innerHTML = "";
  data.sources.forEach((s) => {
    box.insertAdjacentHTML(
      "beforeend",
      `<label>${s.name} 模拟延迟(虚拟时间/次)
        <input type="number" step="0.5" data-source="${s.name}"
               class="lat-input" value="0"></label>`
    );
  });
}

function renderMutateSources(data) {
  const sel = $("mutate-source");
  sel.innerHTML = data.sources
    .map((s) => `<option value="${s.name}">${s.name}（当前 v${s.version}）</option>`)
    .join("");
}

function resetOutputs() {
  stepSession = null;
  $("btn-step-next").disabled = true;
  $("btn-step-run").disabled = true;
  $("step-status").textContent = "";
  $("stop-evidence").className = "evidence empty";
  $("stop-evidence").textContent = "尚未执行";
  $("correctness").className = "evidence empty";
  $("correctness").textContent = "尚未核对";
  $("ranking").querySelector("tbody").innerHTML = "";
  $("ledger").querySelector("tbody").innerHTML = "";
  $("fullcost").querySelector("tbody").innerHTML = "";
  $("candidates").querySelector("tbody").innerHTML = "";
  $("savings").innerHTML = "";
  $("step-event").className = "eventbox empty";
  $("step-event").textContent = "步进事件将显示在这里";
  $("naive").className = "evidence empty";
  $("naive").textContent = "点击「全量关联对照」生成";
  $("mutate-log").className = "eventbox empty";
  $("mutate-log").textContent = "";
}

// ---------------------------------------------------------------- 执行查询

async function doRun() {
  const data = await api("/api/run", configBody());
  const { result, baseline_topk, correctness, full_scan_cost, savings } = data;

  const ev = $("stop-evidence");
  ev.className = "evidence";
  const se = result.stop_evidence || {};
  ev.innerHTML = `
    <div><span class="rule">停止规则：${se.rule || "—"}</span>
      （第 ${result.rounds} 轮停止，模式 ${result.mode.toUpperCase()}，重启 ${result.restarts} 次）</div>
    <div>${se.text || ""}</div>
    <div class="desc">阈值: ${JSON.stringify(se.thresholds || {})}
      ${se.unseen_ub !== undefined ? `· 未见上界 UB=${fmt(se.unseen_ub)}` : ""}
      ${se.boundary_key ? `· 边界=${se.boundary_key}(${fmt(se.boundary_score)})` : ""}</div>`;

  renderLedger(result.ledger, full_scan_cost, savings);
  renderRanking(result.ranking, baseline_topk, correctness);
}

function renderLedger(ledger, fullCost, savings) {
  const names = Object.keys(ledger.sa_count).concat(
    Object.keys(fullCost.sa_count).filter((n) => !(n in ledger.sa_count))
  );
  $("ledger").querySelector("tbody").innerHTML = names
    .map(
      (n) => `<tr><td>${n}</td>
        <td class="num">${ledger.sa_count[n] || 0}</td>
        <td class="num">${ledger.ra_count[n] || 0}</td>
        <td class="num">${ledger.raw_rows_touched[n] || 0}</td>
        <td class="num">—</td></tr>`
    )
    .join("") +
    `<tr><td><b>合计</b></td>
      <td class="num"><b>${ledger.total_sa}</b></td>
      <td class="num"><b>${ledger.total_ra}</b></td>
      <td class="num"><b>${Object.values(ledger.raw_rows_touched).reduce((a, b) => a + b, 0)}</b></td>
      <td class="num"><b>${fmt(ledger.virtual_time)}</b></td></tr>`;

  $("fullcost").querySelector("tbody").innerHTML = Object.keys(fullCost.sa_count)
    .map(
      (n) => `<tr><td>${n}</td>
        <td class="num">${fullCost.sa_count[n]}</td>
        <td class="num">0</td>
        <td class="num">${fullCost.raw_rows_touched[n]}</td>
        <td class="num">—</td></tr>`
    )
    .join("") +
    `<tr><td><b>合计</b></td>
      <td class="num"><b>${fullCost.total_sa}</b></td>
      <td class="num"><b>0</b></td>
      <td class="num"><b>${Object.values(fullCost.raw_rows_touched).reduce((a, b) => a + b, 0)}</b></td>
      <td class="num"><b>${fmt(fullCost.virtual_time)}</b></td></tr>`;

  $("savings").innerHTML =
    `SA 节省 <b>${savings.sa_saved}</b> 次（${savings.sa_saved_pct}%）· ` +
    `虚拟时间节省 <b>${savings.time_saved_pct}%</b>`;
}

function renderRanking(ranking, baseline, correctness) {
  const baseMap = Object.fromEntries(baseline.map((r) => [r.key, r.score]));
  const tb = $("ranking").querySelector("tbody");
  tb.innerHTML = ranking
    .map((r, i) => {
      const base = baseMap[r.key];
      const ok = base !== undefined && Math.abs(base - r.score) < 1e-12;
      return `<tr><td>${i + 1}</td><td>${r.key}</td>
        <td class="num">${fmt(r.score)}</td>
        <td class="num">${base === undefined ? "—" : fmt(base)}</td>
        <td>${ok ? '<span class="tag ok">一致</span>' : '<span class="tag miss">差异</span>'}</td></tr>`;
    })
    .join("");
  const c = $("correctness");
  c.className = "evidence " + (correctness.exact_match ? "good" : "bad");
  c.innerHTML = correctness.exact_match
    ? `<span class="tag ok">精确一致</span> 执行器前 ${correctness.engine_n} 名与全量关联逐名核对完全相同（键序与分数）`
    : `<span class="tag miss">存在差异</span> ${JSON.stringify(correctness.mismatches)}`;
}

// ---------------------------------------------------------------- 全量对照

async function doBaseline() {
  const data = await api("/api/baseline", configBody());
  const box = $("naive");
  box.className = "evidence";
  const missed = data.naive_missed_winners;
  box.innerHTML = `
    <div>全量对象数：<b>${data.n_objects}</b></div>
    <div>朴素局部 TopK 榜单：${data.naive_local_topk.map((r) => r.key).join(", ")}</div>
    <div style="margin-top:8px">
      ${missed.length
        ? `<span class="tag miss">反例成立</span> 朴素拼接漏掉真正赢家：` +
          missed.map((m) => `<b>${m.key}</b>（真实名次 #${m.rank}, 分数 ${fmt(m.score)}）`).join("；")
        : '<span class="tag ok">本数据集下未漏</span>'}
    </div>`;
}

// ---------------------------------------------------------------- 步进

async function doStepStart() {
  const data = await api("/api/step/start", configBody());
  stepSession = data.session_id;
  $("btn-step-next").disabled = false;
  $("btn-step-run").disabled = false;
  $("step-status").textContent =
    `会话 ${stepSession} · 模式 ${data.mode.toUpperCase()} · K=${data.k}`;
  $("step-event").className = "eventbox";
  $("step-event").textContent = "会话已创建，点击「推进一步」";
  $("candidates").querySelector("tbody").innerHTML = "";
}

async function doStepNext() {
  if (!stepSession) return;
  const data = await api("/api/step/next", { session_id: stepSession });
  if (data.finished && !data.event) {
    $("step-status").textContent += " · 已结束";
    $("btn-step-next").disabled = true;
    return;
  }
  renderStep(data);
  if (data.finished) {
    $("btn-step-next").disabled = true;
    if (data.stop_evidence) {
      const ev = $("stop-evidence");
      ev.className = "evidence";
      const se = data.stop_evidence;
      ev.innerHTML = `<div><span class="rule">停止规则：${se.rule}</span>
        （第 ${se.round} 轮）</div><div>${se.text || ""}</div>`;
    }
  }
}

function renderStep(data) {
  const ev = data.event;
  const lines = [];
  lines.push(`# 第 ${ev.round} 轮 · ${ev.mode.toUpperCase()}`);
  if (ev.drift) {
    lines.push(`⚠ 版本漂移事件: ${JSON.stringify(ev.drift.drifted)} — ${ev.drift.note}`);
  }
  if (ev.action === "sa") {
    lines.push(
      `SA 读取源 [${ev.source}] → key=${ev.entry.key}, score=${fmt(ev.entry.score)}, ` +
      `该键在本源 ${ev.entry.n_rows} 行`
    );
    if (ev.random_access) {
      ev.random_access.forEach((ra) =>
        lines.push(
          `  RA 源 [${ra.source}] key=${ev.entry.key} → ${ra.found ? "命中 " + fmt(ra.score) : "确认缺席"}`
        )
      );
    }
    lines.push(`当前阈值: ${JSON.stringify(ev.thresholds)}`);
    lines.push(`任何未见对象可达上界 UB_unseen = ${fmt(ev.unseen_ub)}`);
  } else {
    lines.push("所有源排序访问已耗尽");
  }
  if (ev.stop) lines.push(`■ 停止: ${ev.stop.rule} — ${ev.stop.text || ""}`);
  $("step-event").textContent = lines.join("\n");
  $("step-status").textContent =
    `会话 ${stepSession} · 轮次 ${ev.round} · SA=${data.ledger.total_sa} RA=${data.ledger.total_ra}` +
    (data.finished ? " · 已结束" : "");

  const tb = $("candidates").querySelector("tbody");
  tb.innerHTML = ev.candidates
    .map(
      (c) => `<tr><td>${c.key}</td>
        <td class="num">${fmt(c.lb)}</td>
        <td class="num">${fmt(c.ub)}</td>
        <td>${Object.entries(c.known).map(([s, v]) => `${s}=${fmt(v)}`).join("<br>")}</td>
        <td>${c.absent.join(", ") || "—"}</td>
        <td>${c.unknown.join(", ") || "—"}</td>
        <td class="num">${c.final === null || c.final === undefined ? "未定" : fmt(c.final)}</td></tr>`
    )
    .join("");
}

async function doStepRun() {
  $("btn-step-next").disabled = true;
  // 连续推进直到结束（服务端逐轮产生真实读取，不跳步）
  for (;;) {
    const data = await api("/api/step/next", { session_id: stepSession });
    if (data.finished && !data.event) break;
    renderStep(data);
    if (data.finished) break;
  }
  $("btn-step-run").disabled = true;
}

// ---------------------------------------------------------------- 版本突变

async function doMutate() {
  const source = $("mutate-source").value;
  const data = await api("/api/mutate", { source });
  const box = $("mutate-log");
  box.className = "eventbox";
  box.textContent =
    `源 ${data.source} 版本 v${data.old_version} → v${data.new_version}\n` +
    `受影响进行中会话: ${JSON.stringify(data.active_sessions, null, 2)}\n` +
    data.note + "\n（继续步进即可观察 drift_continue / drift_restart 事件）";
  // 刷新版本号显示
  const meta = await (await fetch("/api/datasets")).json();
  const cur = meta.datasets.find((d) => d.name === $("dataset").value);
  renderMutateSources({ sources: meta.current ? meta.current.sources : cur.sources });
}

// ---------------------------------------------------------------- 绑定

$("btn-load").addEventListener("click", () => doLoad().catch(alertErr));
$("btn-run").addEventListener("click", () => doRun().catch(alertErr));
$("btn-baseline").addEventListener("click", () => doBaseline().catch(alertErr));
$("btn-step-start").addEventListener("click", () => doStepStart().catch(alertErr));
$("btn-step-next").addEventListener("click", () => doStepNext().catch(alertErr));
$("btn-step-run").addEventListener("click", () => doStepRun().catch(alertErr));
$("btn-mutate").addEventListener("click", () => doMutate().catch(alertErr));
// 权重等配置改动后无需手动重载，执行/步进时随请求提交

function alertErr(err) {
  console.error(err);
  alert(err.message);
}

loadDatasets().catch(alertErr);
