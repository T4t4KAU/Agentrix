const $ = (id) => document.getElementById(id);
const state = { phase: "idle", previousTokens: { left: 0, right: 0 } };
let socket;

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.onopen = () => setConnection("connected", "Live controller");
  socket.onclose = () => {
    setConnection("disconnected", "Reconnecting");
    setTimeout(connect, 900);
  };
  socket.onmessage = (event) => render(JSON.parse(event.data));
}

function setConnection(kind, copy) {
  const element = $("connection");
  element.className = `connection ${kind}`;
  element.querySelector("span").textContent = copy;
}

function compact(value) {
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value || 0);
}

function render(payload) {
  state.phase = payload.phase;
  const affinityOwners = buildAffinityOwners(payload.left.routes);
  renderSide("left", payload.left, affinityOwners);
  renderSide("right", payload.right, affinityOwners);
  renderPhase(payload);
}

function renderSide(side, data, affinityOwners) {
  const tokens = $( `${side}-tokens` );
  tokens.textContent = compact(data.tokens);
  if (data.tokens > state.previousTokens[side]) {
    tokens.classList.remove("token-pulse");
    void tokens.offsetWidth;
    tokens.classList.add("token-pulse");
  }
  state.previousTokens[side] = data.tokens;
  $(`${side}-estimate`).textContent = data.tokens_estimated ? "live" : "exact";
  $(`${side}-done`).textContent = `${data.completed_requests} / ${data.expected_requests}`;
  $(`${side}-prefill`).textContent = data.prefill_requests;
  $(`${side}-generating`).textContent = data.generating_requests;
  $(`${side}-waiting`).textContent = data.queued_requests;
  $(`${side}-rate`).textContent = `${data.output_tokens_per_s.toFixed(1)} tok/s`;
  $(`${side}-queue-depth`).textContent = data.waiting_sequences;
  $(`${side}-queue-time`).textContent = data.average_queue_time_ms == null
    ? "—"
    : `${data.average_queue_time_ms.toFixed(0)} ms`;
  $(`${side}-prefill-compute`).textContent = `${compact(data.computed_prefill_tokens)} tokens`;
  $(`${side}-title`).textContent = data.title.split("·").pop().trim();

  if (data.hbm_used_mib.length) {
    const used = average(data.hbm_used_mib) / 1024;
    const total = average(data.hbm_total_mib) / 1024;
    $(`${side}-hbm`).textContent = `${used.toFixed(1)} / ${total.toFixed(1)} GiB`;
    $(`${side}-memory-bar`).style.width = `${Math.min(100, used / total * 100)}%`;
  }
  $(`${side}-kv`).textContent = data.kv_usage.length
    ? `${(average(data.kv_usage) * 100).toFixed(1)}%`
    : "—";
  renderRanks(side, data.rank_loads);
  renderRoutes(side, data.routes, affinityOwners);
  renderStreams(side, data.streams);
}

function buildAffinityOwners(routes) {
  const votes = new Map();
  for (const route of routes) {
    if (!votes.has(route.prefix_group)) votes.set(route.prefix_group, [0, 0, 0, 0]);
    votes.get(route.prefix_group)[route.dp_rank] += 1;
  }
  return new Map(Array.from(votes, ([prefix, counts]) => [
    prefix,
    counts.indexOf(Math.max(...counts)),
  ]));
}

function renderRoutes(side, routes, affinityOwners) {
  const target = $(`${side}-route-groups`);
  if (!routes.length) return;
  const fragment = document.createDocumentFragment();
  for (let rank = 0; rank < 4; rank += 1) {
    const row = document.createElement("div");
    row.className = `route-rank-row rank-${rank}`;
    const label = document.createElement("span");
    label.className = "route-rank-label";
    label.textContent = `rank ${rank}`;
    const sequences = document.createElement("div");
    sequences.className = "route-rank-sequences";
    for (const route of routes.filter((item) => item.dp_rank === rank)) {
      const dot = document.createElement("i");
      const affinityOwner = affinityOwners.get(route.prefix_group);
      const affinityClass = affinityOwner == null ? "affinity-unknown" : `affinity-${affinityOwner}`;
      dot.className = `route-dot ${affinityClass} ${route.status}`;
      dot.title = `${shortCase(route.prefix_group)} · agent-${String(route.agent_id).padStart(2, "0")} → rank ${route.dp_rank} · affinity ${affinityOwner ?? "?"} · ${route.status}`;
      sequences.append(dot);
    }
    row.append(label, sequences);
    fragment.append(row);
  }
  target.replaceChildren(fragment);
}

function renderRanks(side, loads) {
  const target = $(`${side}-ranks`);
  if (!loads.length) return;
  target.replaceChildren(...loads.map(([running, waiting], rank) => {
    const item = document.createElement("span");
    if (running) item.className = "hot";
    item.textContent = `DP${rank} · ${running} running · ${waiting} queued`;
    return item;
  }));
}

function renderStreams(side, streams) {
  const target = $(`${side}-streams`);
  if (!streams.length) return;
  const existing = new Map(
    Array.from(target.querySelectorAll(".agent-card[data-request-id]"))
      .map((card) => [card.dataset.requestId, card]),
  );
  const desired = [];
  for (const stream of streams) {
    let card = existing.get(stream.request_id);
    if (!card) {
      card = createStreamCard(stream.request_id);
    }
    existing.delete(stream.request_id);
    card.className = `agent-card ${stream.status}`;
    const identity = card.querySelector(".agent-id");
    identity.textContent = `${shortCase(stream.case_id)} · agent-${String(stream.agent_id).padStart(2, "0")}`;
    const statusCopy = stream.status === "prefill" ? "prefill / scheduling" : stream.status;
    card.querySelector(".agent-status-copy").textContent = statusCopy;
    const copy = card.querySelector(".agent-output-copy");
    const nextText = tailText(stream.text) || "Waiting for the first token…";
    if (copy.textContent !== nextText) {
      copy.textContent = nextText;
    }
    card.querySelector(".cursor").classList.toggle("hidden", stream.status !== "generating");
    desired.push(card);
  }
  target.querySelector(".empty-state")?.remove();
  desired.forEach((card, index) => {
    if (target.children[index] !== card) {
      target.insertBefore(card, target.children[index] || null);
    }
  });
  for (const card of existing.values()) {
    card.remove();
  }
}

function createStreamCard(requestId) {
  const card = document.createElement("section");
  card.dataset.requestId = requestId;
  const meta = document.createElement("div");
  meta.className = "agent-meta";
  const identity = document.createElement("span");
  identity.className = "agent-id";
  const status = document.createElement("span");
  status.className = "agent-status";
  const statusDot = document.createElement("i");
  const statusCopy = document.createElement("span");
  statusCopy.className = "agent-status-copy";
  status.append(statusDot, statusCopy);
  meta.append(identity, status);
  const output = document.createElement("pre");
  output.className = "agent-output";
  const copy = document.createElement("span");
  copy.className = "agent-output-copy";
  const cursor = document.createElement("span");
  cursor.className = "cursor hidden";
  output.append(copy, cursor);
  card.append(meta, output);
  return card;
}

function renderPhase(payload) {
  const button = $("start");
  const phaseCopy = $("phase-copy");
  const copy = {
    idle: "Waiting to start",
    preparing: "Establishing identical shared-prefix cohorts…",
    running: "Both systems are generating live",
    completed: "Run complete · exact usage finalized",
    error: `Run failed · ${payload.error || "inspect controller log"}`,
  };
  phaseCopy.textContent = copy[payload.phase] || payload.phase;
  button.disabled = ["preparing", "running"].includes(payload.phase);
  button.querySelector("span").textContent = payload.phase === "completed" ? "Run again" : "Start fair run";

  const speedup = $("speedup");
  if (payload.speedup && payload.phase === "running") {
    speedup.classList.remove("hidden");
    speedup.querySelector("strong").textContent = `${payload.speedup.toFixed(2)}×`;
  } else if (payload.phase !== "completed") {
    speedup.classList.add("hidden");
  }
  if (payload.speedup && payload.phase === "completed") {
    speedup.classList.remove("hidden");
    speedup.querySelector("strong").textContent = `${payload.speedup.toFixed(2)}×`;
    speedup.querySelector("span").textContent = "final throughput";
  }
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function shortCase(value) {
  return value.length > 30 ? `${value.slice(0, 27)}…` : value;
}

function tailText(value) {
  if (value.length <= 900) return value;
  return `…${value.slice(-899)}`;
}

$("start").addEventListener("click", () => {
  if (socket?.readyState === WebSocket.OPEN) socket.send("start");
});

connect();
